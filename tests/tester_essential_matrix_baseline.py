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
from functions import point_matching, normalize_keypoints, SuperPointWithAffNetScaleNetKornia
from evaluation import evaluate_R_t, pose_auc
from lightglue import LightGlue, SuperPoint
from romatch import roma_outdoor
import kornia as K
import kornia.feature as KF

import pygcransac
import pymagsac
import poselib
import pycolmap
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from datasets.scannet import ScanNet
from datasets.lamar import Lamar
from datasets.eth3d import ETH3D
from datasets.kitti import Kitti
from datasets.phototourism import PhotoTourism
from datasets.seven_scenes import SevenScenes

def run(lafs1, lafs2, matches, scores, K1, K2, R_gt, t_gt, image_size1, image_size2, args):
    # Initialize the errors
    error = 1e10
    rotation_error = 1e10
    translation_error = 1e10
    absolute_translation_error = 1e10
    
    # Reshape the LAFs from 1 * n * 2 * 3 to n * 6
    lafs1 = lafs1.reshape(-1, 6)
    lafs2 = lafs2.reshape(-1, 6)

    # Remove rows from the matches where the first value is -1
    mask = matches[:, 0] != -1
    matches = matches[mask, :]

    # Swap the columns of the LAFs so that the coordinates comes first
    pts1 = lafs1[:, [2, 5]]
    pts1 = pts1[mask, :]
    pts2 = lafs2[:, [2, 5]]
    
    correspondences = np.zeros((pts1.shape[0], 4))
    correspondences[:, :2] = pts1
    correspondences[:, 2:] = pts2[matches[:, 0].astype(int), :]

    # Return if there are fewer than 4 correspondences
    if correspondences.shape[0] < 8:
        return (np.inf, np.inf), 0, 0

    if args.method == "poselib":
        ransac_options = {"max_iterations": args.maximum_iterations,
                        "min_iterations":  args.minimum_iterations,
                        "success_prob": args.confidence,
                        "max_epipolar_error": args.inlier_threshold,
                        "progressive_sampling": True}
        
        tic = time.perf_counter()
        F_est, res  = poselib.estimate_fundamental(correspondences[:,:2], correspondences[:,2:4], ransac_opt=ransac_options)
        inliers = np.array(res['inliers'])
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "pycolmap":
        ransac_options = pycolmap.RANSACOptions(
            max_error=args.inlier_threshold,  # for example the reprojection error in pixels
            min_inlier_ratio=0.01,
            confidence=0.9999,
            min_num_trials=args.minimum_iterations,
            max_num_trials=args.maximum_iterations,
        )
        
        xy1 = np.ascontiguousarray(correspondences[:, 0:2]).astype(np.float64)
        xy2 = np.ascontiguousarray(correspondences[:, 2:4]).astype(np.float64)

        tic = time.perf_counter()
        res  = pycolmap.fundamental_matrix_estimation(xy1, xy2, estimation_options = ransac_options)
        F_est = res['F']
        inliers = np.array(res['inliers'])
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "RANSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        F_est, inliers = cv2.findFundamentalMat(
            np.ascontiguousarray(correspondences[:, :2]),
            np.ascontiguousarray(correspondences[:, 2:4]),
            cv2.RANSAC,
            ransacReprojThreshold = args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "LMEDS OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        F_est, inliers = cv2.findFundamentalMat(
            np.ascontiguousarray(correspondences[:, :2]),
            np.ascontiguousarray(correspondences[:, 2:4]),
            cv2.LMEDS,
            ransacReprojThreshold = args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "skimage":
        if correspondences.shape[0] < 9:
            return (np.inf, np.inf), 0, 0
        tic = time.perf_counter()
        model, inliers = ransac(
            (correspondences[:, :2], correspondences[:, 2:4]),
            FundamentalMatrixTransform,
            min_samples = 8,
            residual_threshold = args.inlier_threshold,
            max_trials = args.maximum_iterations,
        )
        F_est = model.params
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "gcransac":        
        tic = time.perf_counter()
        F_est, inliers = pygcransac.findFundamentalMatrix(
            np.ascontiguousarray(correspondences),
            int(K1[1, 2] * 2),
            int(K1[0, 2] * 2),
            int(K2[1, 2] * 2),
            int(K2[0, 2] * 2),
            threshold = args.inlier_threshold,
            sampler = 1,
            max_iters = args.maximum_iterations,
            min_iters = args.maximum_iterations,
            use_sprt = True,
            probabilities=[],
        )
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "magsac":        
        tic = time.perf_counter()
        F_est, inliers = pymagsac.findFundamentalMatrix(
            np.ascontiguousarray(correspondences), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            use_magsac_plus_plus = False,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "magsac++":        
        tic = time.perf_counter()
        F_est, inliers = pymagsac.findFundamentalMatrix(
            np.ascontiguousarray(correspondences), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            use_magsac_plus_plus = True,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic

    if F_est is None:
        return (np.inf, np.inf), 0, elapsed_time

    #print(F_est)

    # Convert the fundamental matrix to essential matrix if the estimation is successful
    E_est = K2.T @ F_est @ K1

    norm_correspondences = np.zeros(correspondences.shape)
    norm_correspondences[:, :2] = normalize_keypoints(correspondences[:, :2], K1)
    norm_correspondences[:, 2:] = normalize_keypoints(correspondences[:, 2:], K2)

    # Decompose the essential matrix to get the relative pose
    _, R, t, _ = cv2.recoverPose(E_est, norm_correspondences[inliers, :2], norm_correspondences[inliers, 2:])
    
    # Count the inliers
    inlier_number = inliers.sum()

    return evaluate_R_t(R_gt, t_gt, R, t), inlier_number, elapsed_time

if __name__ == "__main__":
    # Passing the arguments
    parser = argparse.ArgumentParser(description="Running on the baselines on the essential matrix estimation datasets.")
    parser.add_argument('--batch_size', type=int, help="Batch size for multi-CPU processing", default=1000)
    parser.add_argument('--root_dir_scannet', type=str, help="The path to where the ScanNet dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test")
    parser.add_argument('--root_dir_phototourism', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/RANSAC-Tutorial-Data")
    parser.add_argument('--root_dir_lamar', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/lamar/CAB/sessions/query_val_hololens")
    parser.add_argument('--root_dir_eth3d', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/eth3d")
    parser.add_argument('--root_dir_7scenes', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd2tb/datasets/7scenes")
    parser.add_argument('--root_dir_kitti', type=str, help="The path to where the dataset is located and the matches should be saved.", default="/media/hdd3tb/datasets/kitti/dataset")
    parser.add_argument("--results_path", type=str, default="results_test_essential_matrix_baselines.csv")
    parser.add_argument("--confidence", type=float, default=0.9999999)
    parser.add_argument("--upright", type=bool, default=True)
    parser.add_argument("--core_number", type=int, default=19)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
        
    # Thresholds for the different methods set by tuning on random 200 pairs from each dataset
    thresholds = {
        "poselib": 3.0, 
        "pycolmap": 1.5, 
        "RANSAC OpenCV": 0.75, 
        "gcransac": 1.0, 
        "magsac": 3.0, 
        "magsac++": 5.0
    }

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
                for method in ["poselib", "pycolmap", "RANSAC OpenCV", "gcransac", "magsac", "magsac++"]:
                    for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                        threshold = thresholds[method]

                        # Select a random subset of the tests
                        key = (threshold, method, iters)
                        if key not in pose_errors:
                            pose_errors[key] = []
                            runtimes[key] = []
                            inlier_numbers[key] = []

                        args.method = method
                        args.inlier_threshold = threshold
                        args.scene_idx = i
                        args.maximum_iterations = iters
                        args.minimum_iterations = iters

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
            for method in ["poselib", "pycolmap", "RANSAC OpenCV", "gcransac", "magsac", "magsac++"]:
                for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                    args.maximum_iterations = iters
                    args.minimum_iterations = iters
                    threshold = thresholds[method]

                    key = (threshold, method, iters)
                    curr_pose_errors = np.array(pose_errors[key])
                    auc_R = 100 * np.r_[pose_auc(curr_pose_errors[:,0], thresholds=[5, 10, 20])]
                    auc_t = 100 * np.r_[pose_auc(curr_pose_errors[:,1], thresholds=[5, 10, 20])]
                    auc_Rt = 100 * np.r_[pose_auc(curr_pose_errors.max(1), thresholds=[5, 10, 20])]

                    # Remove inf values
                    curr_pose_errors = curr_pose_errors[np.isfinite(curr_pose_errors).all(axis=1)]
                    f.write(f"{method},splg+affnet,{db_name},{threshold},{iters},{auc_R[0]},{auc_R[1]},{auc_R[2]},{auc_t[0]},{auc_t[1]},{auc_t[2]},{auc_Rt[0]},{auc_Rt[1]},{auc_Rt[2]},{np.mean(curr_pose_errors)},{np.median(curr_pose_errors)},{np.mean(runtimes[key])},{np.median(runtimes[key])},{np.mean(inlier_numbers[key])},{np.median(inlier_numbers[key])}\n")
