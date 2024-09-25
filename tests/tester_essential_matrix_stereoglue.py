import cv2
import h5py
import os
import numpy as np
import argparse
import yaml
import math
import time
from utils import detect_and_load_data
from joblib import Parallel, delayed
from tqdm import tqdm
from random import sample
from functions import point_matching, normalize_keypoints, SuperPointWithAffNetScaleNetKornia
from evaluation import evaluate_R_t, pose_auc
from lightglue import LightGlue, SuperPoint
from romatch import roma_outdoor
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

from datasets.scannet import ScanNet
from datasets.lamar import Lamar
from datasets.eth3d import ETH3D
from datasets.kitti import Kitti
from datasets.phototourism import PhotoTourism
from datasets.seven_scenes import SevenScenes

import pystereoglue

def run(lafs1, lafs2, matches, scores, K1, K2, R_gt, t_gt, image_size1, image_size2, args):
    # Return if there are fewer than 5 correspondences
    if matches.shape[0] < 5:
        return (np.inf, np.inf), 0, 0
    
    # Set up the configuration
    config = pystereoglue.RANSACSettings()
    config.inlier_threshold = args.inlier_threshold
    config.max_iterations = args.maximum_iterations
    config.confidence = args.confidence
    config.core_number = 8
        
    if args.scoring == "RANSAC":
        config.scoring = pystereoglue.ScoringType.RANSAC
    elif args.scoring == "MSAC":
        config.scoring = pystereoglue.ScoringType.MSAC
    elif args.scoring == "MAGSAC":
        config.scoring = pystereoglue.ScoringType.MAGSAC
        
    if args.lo == "LSQ":
        config.local_optimization = pystereoglue.LocalOptimizationType.LSQ
    elif args.lo == "IRLS":
        config.local_optimization = pystereoglue.LocalOptimizationType.IteratedLSQ
    elif args.lo == "NestedRANSAC":
        config.local_optimization = pystereoglue.LocalOptimizationType.NestedRANSAC
    elif args.lo == "Nothing":
        config.local_optimization = pystereoglue.LocalOptimizationType.Nothing
        
    if args.fo == "LSQ":
        config.final_optimization = pystereoglue.LocalOptimizationType.LSQ
    elif args.fo == "IRLS":
        config.final_optimization = pystereoglue.LocalOptimizationType.IteratedLSQ
    elif args.fo == "NestedRANSAC":
        config.final_optimization = pystereoglue.LocalOptimizationType.NestedRANSAC
    elif args.fo == "Nothing":
        config.final_optimization = pystereoglue.LocalOptimizationType.Nothing

    matches = matches[:, :args.pool_size]
    scores = scores[:, :args.pool_size]

    # Reshape the LAFs from 1 * n * 2 * 3 to n * 6
    lafs1 = lafs1.reshape(-1, 6)
    lafs2 = lafs2.reshape(-1, 6)

    # Remove rows from the matches where the first value is -1
    mask = matches[:, 0] != -1
    matches = matches[mask, :]
    scores = scores[mask]
    lafs1 = lafs1[mask, :]

    # Swap the columns of the LAFs so that the coordinates comes first
    lafs1 = lafs1[:, [2, 5, 0, 1, 3, 4]]
    lafs2 = lafs2[:, [2, 5, 0, 1, 3, 4]]

    if lafs1.shape[0] >= 6:
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        E_est, inliers, score, iterations = pystereoglue.estimateEssentialMatrixGravity(
            np.ascontiguousarray(lafs1),
            np.ascontiguousarray(lafs2),
            np.ascontiguousarray(matches), 
            np.ascontiguousarray(scores), 
            K1,
            K2,
            [image_size1[2], image_size1[1], image_size2[2], image_size2[1]],
            gravity_src = np.identity(3),
            gravity_dst = np.identity(3),
            config = config)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    else: 
        return (np.inf, np.inf), 0, 0.0

    if E_est is None:
        return (np.inf, np.inf), 0, elapsed_time
    
    try:
        # Count the inliers
        inlier_number = len(inliers)

        # Convert the inliers to a numpy array
        inliers = np.array(inliers)
        
        # Compose the inlier correspondences from the LAFs
        correspondences = np.zeros((inliers.shape[0], 4))
        correspondences[:, :2] = lafs1[inliers[:, 0].astype(np.int32), :2]
        correspondences[:, 2:] = lafs2[inliers[:, 1].astype(np.int32), :2]
        
        # Normalize the correspondences
        norm_correspondences = np.zeros((correspondences.shape[0], 4))
        norm_correspondences[:, :2] = normalize_keypoints(correspondences[:, :2], K1)
        norm_correspondences[:, 2:] = normalize_keypoints(correspondences[:, 2:4], K2)
        
        # Decompose the essential matrix to get the relative pose
        if len(inliers) > 0:
            _, R, t, _ = cv2.recoverPose(E_est, norm_correspondences[:, :2], norm_correspondences[:, 2:])
        else:
            R = np.eye(3)
            t = np.zeros((3, 1))

        return evaluate_R_t(R_gt, t_gt, R, t), inlier_number, elapsed_time
    except:
        return (np.inf, np.inf), 0, elapsed_time

if __name__ == "__main__":
    # Passing the arguments
    parser = argparse.ArgumentParser(description="Running on the HEB benchmark")
    parser.add_argument('--batch_size', type=int, help="Batch size for multi-CPU processing", default=1000)
    parser.add_argument('--output_db_path', type=str, help="The path to where the dataset of matches should be saved.", default="/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test/matches.h5")
    parser.add_argument('--root_dir_scannet', type=str, help="The path to where the ScanNet dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test")
    parser.add_argument('--root_dir_phototourism', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/RANSAC-Tutorial-Data")
    parser.add_argument('--root_dir_lamar', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/lamar/CAB/sessions/query_val_hololens")
    parser.add_argument('--root_dir_eth3d', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/eth3d")
    parser.add_argument('--root_dir_7scenes', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/7scenes")
    parser.add_argument('--root_dir_kitti', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/kitti/dataset")
    parser.add_argument("--results_path", type=str, default="results_test_essential_matrix_stereoglue.csv")
    parser.add_argument("--inlier_threshold", type=float, default=5.0)
    parser.add_argument("--maximum_iterations", type=int, default=1000)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--upright", type=bool, default=True)
    parser.add_argument("--scoring", type=str, help="Choose from: RANSAC, MSAC, MAGSAC.", choices=["RANSAC", "MSAC", "MAGSAC"], default="MAGSAC")
    parser.add_argument("--lo", type=str, help="Choose from: LSQ, IRLS, NestedRANSAC, Nothing.", choices=["LSQ", "IRLS", "NestedRANSAC", "Nothing"], default="NestedRANSAC")
    parser.add_argument("--fo", type=str, help="Choose from: LSQ, IRLS, NestedRANSAC, Nothing.", choices=["LSQ", "IRLS", "NestedRANSAC", "Nothing"], default="LSQ")
    parser.add_argument("--core_number", type=int, default=18)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    
    print(f"Tuning StereoGlue")
    
    datasets = [ScanNet(root_dir=os.path.expanduser(args.root_dir_scannet), split='test'),
                PhotoTourism(root_dir=os.path.expanduser(args.root_dir_phototourism), split='val'),
                Lamar(root_dir=os.path.expanduser(args.root_dir_lamar)),
                ETH3D(root_dir=os.path.expanduser(args.root_dir_eth3d), split='test', downsize_factor=8),
                SevenScenes(root_dir=os.path.expanduser(args.root_dir_7scenes), split='test', scene='all'),
                Kitti(root_dir=args.root_dir_kitti, steps=20)]

    print("Initialize SP+LG detector")
    detector = SuperPointWithAffNetScaleNetKornia(init_scale = 20, upright = args.upright)  # load the extractor
    matcher = KF.LightGlue('superpoint', width_confidence=-1, depth_confidence=-1).eval().to(args.device)
    
    for dataset in datasets:
        pose_errors = {}
        runtimes = {}
        inlier_numbers = {}
        db_name = dataset.__class__.__name__.lower()
        dataloader = dataset.get_dataloader()
        args.output_db_path = dataset.root_dir + "/stereoglue_matches.h5"

        processing_queue = []
        run_count = 1
        for i, data in enumerate(dataloader):
            lafs1, lafs2, matches, scores = detect_and_load_data(data, args, detector, matcher, upright = args.upright)
            print(f"Processing pair [{i + 1} / {len(dataloader)}]")
            processing_queue.append((data, lafs1, lafs2, matches, scores))
            
            ## Running the estimators so we don't have too much things in the memory
            if len(processing_queue) >= args.batch_size or i == len(dataloader) - 1:
                for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                    # Select a random subset of the tests
                    key = (iters)
                    if key not in pose_errors:
                        pose_errors[key] = []
                        runtimes[key] = []
                        inlier_numbers[key] = []

                    args.scene_idx = i
                    args.maximum_iterations = iters

                    results = Parallel(n_jobs=min(args.core_number, len(processing_queue)))(delayed(run)(
                        lafs1,
                        lafs2,
                        matches,
                        scores,
                        data["K1"],
                        data["K2"],
                        data["R_1_2"],
                        data["T_1_2"],
                        data["img1"].shape,
                        data["img2"].shape,
                        args) for data, lafs1, lafs2, matches, scores in tqdm(processing_queue))
                    
                    # Concatenating the results to the main lists
                    pose_errors[key] += [error for error, inlier_number, time in results]
                    runtimes[key] += [time for error, inlier_number, time in results]
                    inlier_numbers[key] += [inlier_number for error, inlier_number, time in results]
                    
                # Clearing the processing queue
                processing_queue = []
                run_count += 1

        # Write results into csv
        if not os.path.exists(args.results_path):
            with open(args.results_path, "w") as f:
                f.write("method,features,dataset,threshold,maximum_iterations,auc_R5,auc_R10,auc_R20,auc_t5,auc_t10,auc_t20,auc_Rt5,auc_Rt10,auc_Rt20,avg_error,med_error,avg_time,median_time,avg_inliers,median_inliers\n")
        with open(args.results_path, "a") as f:
            for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                key = (iters)
                curr_pose_errors = np.array(pose_errors[key])
                auc_R = 100 * np.r_[pose_auc(curr_pose_errors[:,0], thresholds=[5, 10, 20])]
                auc_t = 100 * np.r_[pose_auc(curr_pose_errors[:,1], thresholds=[5, 10, 20])]
                auc_Rt = 100 * np.r_[pose_auc(curr_pose_errors.max(1), thresholds=[5, 10, 20])]

                # Remove inf values
                curr_pose_errors = curr_pose_errors[np.isfinite(curr_pose_errors).all(axis=1)]
                f.write(f"stereoglue,splg+affnet,{db_name},{args.inlier_threshold},{iters},{auc_R[0]},{auc_R[1]},{auc_R[2]},{auc_t[0]},{auc_t[1]},{auc_t[2]},{auc_Rt[0]},{auc_Rt[1]},{auc_Rt[2]},{np.mean(curr_pose_errors)},{np.median(curr_pose_errors)},{np.mean(runtimes[key])},{np.median(runtimes[key])},{np.mean(inlier_numbers[key])},{np.median(inlier_numbers[key])}\n")
