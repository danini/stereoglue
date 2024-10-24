import numpy as np
import cv2
import torch
import torch.nn as nn
import time

import kornia as K
import kornia.feature as KF
from lightglue.utils import load_image, rbd
from lightglue import SuperPoint
from scale_affNet_dirty import PatchPoseNet

class SuperPointWithAffNetScaleNetKornia(nn.Module):
    def __init__(self,  upright = True, init_scale=60.0):
        super().__init__()
        self.affnet = KF.LAFAffNetShapeEstimator(True, preserve_orientation=True).eval()
        if upright:
            self.orinet = KF.PassLAF()
        else:
            self.orinet = KF.LAFOrienter(angle_detector=KF.OriNet(True)).eval()
        self.init_scale = float(init_scale)
        self.xy_net = SuperPoint(max_num_keypoints=2048).eval()  # load the extractor
        self.descriptor = KF.LAFDescriptor(KF.HardNet(True)).eval()
        trained_model_name ='_0621_092607_resnet18_ori36_sca13_branchsca-best_model.pt'
        
        ## hyperparameters
        scale_hist_size = int(trained_model_name.split('-')[-2].split('sca')[1].split('_')[0])
        scale_hist_size_one_way = (scale_hist_size -1) / 2

        orient_hist_size = int(trained_model_name.split('-')[-2].split('ori')[1].split('_')[0])
        orient_hist_interval = 360 / orient_hist_size

        backbone = 'resnet' + trained_model_name.split('resnet')[1].split('_')[0]
        self.scale_det = PatchPoseNet(backbone=backbone, output_ori = orient_hist_size, output_sca = scale_hist_size, use_pretrain=False)#
        self.scale_det.load_state_dict(torch.load(trained_model_name, map_location=torch.device('cpu') ))
        self.scale_det.eval()

    def forward(self, x:torch.Tensor, mask=None):
        self.affnet = self.affnet.to(x.device)
        if 'cuda' in str(x.device):
            self.xy_net.cuda = True
        self.scale_det = self.scale_det.to(x.device)
        self.orinet = self.orinet.to(x.device)
        self.descriptor = self.descriptor.to(x.device)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        
        with torch.inference_mode():
            feats = self.xy_net.extract(x)  # auto-resize the image, disable with resize=None
            pts = feats["keypoints"].reshape(-1, 2).float()
            resp = feats["keypoint_scores"].reshape(1, -1, 1)
            desc_sp = feats["descriptors"].reshape(-1, 256)
            lafs_no_scale = KF.laf_from_center_scale_ori(pts.view(1, -1, 2))
            lafs_fixed_scale = KF.scale_laf(lafs_no_scale, self.init_scale)
            patches1 = KF.extract_patches_from_pyramid(x, lafs_fixed_scale)
            scale_dist = self.scale_det(patches1[0])
            scale_coef = torch.pow(2.0, scale_dist[1].argmax(dim=1) * (4./(scale_dist[1].shape[1]-1)))
            lafs = KF.scale_laf(lafs_fixed_scale, scale_coef[None].view(1,-1,1,1))
            new_scale = KF.get_laf_scale(lafs).view(-1)
            good_mask = new_scale < 2000
            lafs = lafs[:, good_mask]
            resp = resp[:, good_mask]
        lafs = self.affnet(lafs, x.mean(dim=1, keepdim=True))
        lafs = self.orinet(lafs, x.mean(dim=1, keepdim=True))
        desc_hardnet = self.descriptor(x.mean(dim=1, keepdim=True), lafs)
        return lafs.cpu().detach().numpy(), resp.cpu().detach().numpy(), desc_sp.cpu().detach().numpy(), desc_hardnet[0].cpu().detach().numpy()

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def find_relative_pose_from_points(kp_matches, K1, K2, kp_scores=None):
    if kp_scores is not None:
        # Select the points with lowest ratio score
        good_matches = kp_scores < 0.8
        pts1 = kp_matches[good_matches, :2]
        pts2 = kp_matches[good_matches, 2:]
    else:
        pts1 = kp_matches[:, :2]
        pts2 = kp_matches[:, 2:]

    if len(pts1) < 5:
        return None, None, None, None

    # Normalize KP
    p1n = normalize_keypoints(pts1, K1)
    p2n = normalize_keypoints(pts2, K2)

    # Find the essential matrix with OpenCV RANSAC
    E, inl_mask = cv2.findEssentialMat(p1n, p2n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None, None, None, None

    # Obtain the corresponding pose
    best_num_inliers = 0
    ret = None
    mask = np.array(inl_mask)[:, 0].astype(bool)
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, p1n, p2n, np.eye(3), 1e9, mask=inl_mask[:, 0])
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], pts1[mask], pts2[mask])
    return ret


def splg_matching(feats0, feats1, matcher, k = 6, threshold = -20.0):
    with torch.inference_mode():
        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        idxs = matches01["matches0"]
        log_assignment = matches01['log_assignment'][0].cpu()
        index_pool = -1 * np.ones((feats0["keypoints"].shape[1], k), dtype=np.int32)
        score_pool = np.zeros((feats0["keypoints"].shape[1], k), dtype=np.float32)
        
        for row in range(idxs.shape[1]):
            if idxs[0, row] == -1:
                continue            
            query_idx = row
            scores = log_assignment[query_idx, :-1]
            scores = np.ravel(scores)
            # Get the indices of the k largest elements
            indices_of_largest = np.argsort(scores)[-k:]
            
            # Reverse the array to get the largest elements first
            indices_of_largest = indices_of_largest[::-1]
            kept_score_indices = scores[indices_of_largest] > threshold
            indices_of_largest = indices_of_largest[kept_score_indices]

            index_pool[query_idx, :len(indices_of_largest)] = indices_of_largest
            score_pool[query_idx, :len(indices_of_largest)] = scores[indices_of_largest]

    return index_pool, score_pool

def point_matching(descs1, descs2, matcher):
    return splg_matching(descs1, descs2, matcher)
