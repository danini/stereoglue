import os, torch, argparse
import torch.nn as nn
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import OrderedDict


import matplotlib.pyplot as plt
import kornia as K
import cv2
import numpy as np
import torch
from kornia_moons.feature import *
import matplotlib as mpl
import kornia.feature as KF



class PatchPoseNet(nn.Module):
    def __init__(self, backbone, output_ori, output_sca, use_pretrain):
        super(PatchPoseNet, self).__init__()

        final_output_channel = None
        if 'resnet' in backbone:
            if backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=use_pretrain)
                final_output_channel = 2048
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=use_pretrain)
                final_output_channel = 2048
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(pretrained=use_pretrain)
                final_output_channel = 512
            elif backbone == 'resnet18':
                self.backbone = models.resnet18(pretrained=use_pretrain)
                final_output_channel = 512
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

            self.orientation_learners = nn.Sequential(  ## classifier
                nn.Linear(final_output_channel, final_output_channel),
                nn.BatchNorm1d(final_output_channel),
                nn.ReLU(),
                nn.Linear(final_output_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_ori)
            )
            self.scale_learners = nn.Sequential(  ## classifier
                nn.Linear(final_output_channel, final_output_channel),
                nn.BatchNorm1d(final_output_channel),
                nn.ReLU(),
                nn.Linear(final_output_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_sca)
            )

        self.GlobalMaxPooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x_ori = self.orientation_learners(self.GlobalMaxPooling(x).view(B, -1))
        x_sca = self.scale_learners(self.GlobalMaxPooling(x).view(B, -1))

        return x_ori, x_sca

    def state_dict(self):
        res = OrderedDict()
        res['backbone'] = self.backbone.state_dict()
        res['orientation_learners'] = self.orientation_learners.state_dict()
        res['scale_learners'] = self.scale_learners.state_dict()

        return res

    def load_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict['backbone'])
        self.orientation_learners.load_state_dict(state_dict['orientation_learners'])
        self.scale_learners.load_state_dict(state_dict['scale_learners'])

    def eval(self):
        self.backbone.eval()
        self.orientation_learners.eval()
        self.scale_learners.eval()

    def train(self):
        self.backbone.train()
        self.orientation_learners.train()
        self.scale_learners.train()





class OpenCVDetectorWithAffNetScaleNetKornia(nn.Module):
    def __init__(self, opencv_detector, scale_det = None, make_upright = False, mrSize: float = 6.0, max_kpts = -1):
        super().__init__()
        self.features = opencv_detector
        self.mrSize = mrSize
        self.affnet = KF.LAFAffNetShapeEstimator(True).eval()
        self.make_upright = make_upright
        self.max_kpts = max_kpts
        self.scale_det = scale_det


    def forward(self, x:torch.Tensor, mask=None):
        self.affnet = self.affnet.to(x.device)
        self.scale_det  = self.scale_det.to(x.device)
        max_val = x.max()
        if max_val < 2.0:
            img_np = (255 * K.tensor_to_image(x)).astype(np.uint8)
        else:
            img_np =  K.tensor_to_image(x).astype(np.uint8)
        if mask is not None:
            mask = K.tensor_to_image(x).astype(np.uint8)
        kpts = self.features.detect(img_np, mask)

            # Compute descriptors
        if self.make_upright:
            unique_kp = []
            for i, kk in enumerate(kpts):
                if i > 0:
                    if kk.response == kpts[i - 1].response:
                        continue
                kk.angle = 0
                unique_kp.append(kk)
            top_resps = np.array([kk.response for kk in unique_kp])
            idxs = np.argsort(top_resps)[::-1]
            if self.max_kpts < 0:
                self.max_kpts = 100000000
            kpts = [unique_kp[i]  for i in idxs[:min(len(unique_kp), self.max_kpts)]]
        lafs, resp = laf_from_opencv_kpts(kpts, mrSize=self.mrSize, with_resp=True, device=x.device)
        with torch.no_grad():
            patches1 = KF.extract_patches_from_pyramid(x, lafs)
            scale_dist = self.scale_det(patches1[0])
            scale_coef = torch.pow(2.0, scale_dist[1].argmax(dim=1) * (4./(scale_dist[1].shape[1]-1)))
            lafs = KF.scale_laf(lafs, scale_coef[None].view(1,-1,1,1))
            new_scale = KF.get_laf_scale(lafs).view(-1)
            good_mask = new_scale < 2000
            lafs = lafs[:, good_mask]
            resp = resp[:, good_mask]

        ori = KF.get_laf_orientation(lafs)
        lafs = self.affnet(lafs, x.mean(dim=1, keepdim=True))
        lafs = KF.set_laf_orientation(lafs, ori)
        return lafs, resp

def get_DoGScaleAffNetHardNet(num_feature = 8000, make_upright = False, mrSize: float = 6.0, max_kpts = -1):
    trained_model_name ='_0621_092607_resnet18_ori36_sca13_branchsca-best_model.pt'
    ## hyperparameters

    scale_hist_size = int(trained_model_name.split('-')[-2].split('sca')[1].split('_')[0])
    scale_hist_size_one_way = (scale_hist_size -1) / 2

    orient_hist_size = int(trained_model_name.split('-')[-2].split('ori')[1].split('_')[0])
    orient_hist_interval = 360 / orient_hist_size

    backbone = 'resnet' + trained_model_name.split('resnet')[1].split('_')[0]
    net = PatchPoseNet(backbone=backbone, output_ori = orient_hist_size, output_sca = scale_hist_size, use_pretrain=False)#.cuda()
    net.load_state_dict(torch.load(trained_model_name, map_location=torch.device('cpu') ))
    net.eval()
    model = OpenCVDetectorWithAffNetScaleNetKornia(cv2.SIFT_create(num_feature), net, make_upright, mrSize, max_kpts)
    feature = KF.LocalFeature(model, KF.LAFDescriptor(KF.HardNet(True)))
    return feature
