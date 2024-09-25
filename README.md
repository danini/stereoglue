# StereoGlue

## Installation

Clone the repository and its submodules:
```
git clone https://github.com/danini/stereoglue.git
```

Make sure that you have the necessary libraries installed:
```
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev libeigen3-dev libboost-all-dev pybind11
```

Install StereoGlue by running
```
pip install .
```

## Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support

# Evaluation

## Jupyter Notebook examples

The example for essential matrix fitting with gravity-based solver is available at: [notebook](examples/example_essential_matrix_fitting_gravity.ipynb).

## Essential Matrix Estimation
To test StereoGlue and other baselines on essential matrix estimation, set up the datasets as described below. 
To run the baselines, use:
```
python tests/tester_essential_matrix_baseline.py
```
To run the StereoGlue, use:
```
python tests/tester_essential_matrix_stereoglue.py
```
The results can finally be plotted by notebook
```
plot-results-essential-matrix.ipynb
```

## Benchmarking results

<p align="center">
  <img src="assets/results_eth3d.png" alt="results_E_eth3d" width="30%" />
  <img src="assets/results_eth3d.png" alt="results_E_phototourism" width="30%" />
  <img src="assets/results_eth3d.png" alt="results_E_lamar" width="30%" />
</p>

## Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf  RANSAC-Tutorial-Data-EF.tar
```

## Evaluation on the ScanNet dataset
Download the data from the test set for relative pose estimation used in SuperGlue (~250Mb for 1500 image pairs only):
```
wget https://www.polybox.ethz.ch/index.php/s/lAZyxm62WUh27Zl/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

## Evaluation on the 7Scenes dataset
Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and put it where it suits you. You can also only download one scene and later specify this scene in the dataloader constructor.

## Evaluation on the ETH3D dataset
Download the [ETH3D dataset](https://www.eth3d.net/datasets) (training split of the high-res multi-view, undistorted images + GT extrinsics & intrinsics should be enough) and put it where it suits you. The input argument 'downsize_factor' can be used to downscale the images, because they can be quite large otherwise.

## Evaluation on the LaMAR dataset
Download the [CAB scene of the LaMAR dataset](https://cvg-data.inf.ethz.ch/lamar/CAB.zip), and unzip it to your favourite location. Note that we only use the images in `CAB/sessions/query_val_hololens`.

## Evaluation on the KITTI dataset
Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images and poses), and unzip them to your favourite location.

# Acknowledgements

When using the algorithm, please cite:
```
@inproceedings{StereoGlue2024,
	author = {Barath, Daniel and Mishkin, Dmytro and Cavalli, Luca and Sarlin, Paul-Edouard and Hruby, Petr and Pollefeys, Marc},
	title = {{StereoGlue}: Robust Estimation with Single-Point Solvers},
	booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2024},
}
```

If you use it for fundamental matrix estimation, please cite:
```
@inproceedings{StereoGlue2024,
	author = {Hruby, Petr and Pollefeys, Marc and Barath, Daniel},
	title = {Semicalibrated Relative Pose from an Affine Correspondence and Monodepth},
	booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2024},
}
```

If you use it with MAGSAC++ scoring, please cite:
```
@inproceedings{barath2020magsac++,
  title={MAGSAC++, a fast, reliable and accurate robust estimator},
  author={Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiri},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1304--1312},
  year={2020}
}
```
