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

To run affine feature detection and matching with the built-in tools, install:
```
pip install kornia
pip install kornia-moons

git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
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
The example for fundamental matrix fitting with monodepth-based solver is available at: [notebook](examples/example_fundamental_matrix_fitting.ipynb).

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
@inproceedings{Hruby2024,
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
