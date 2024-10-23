#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "stereoglue.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "termination/types.h"
#include "local_optimization/types.h"
#include "utils/types.h"
#include "settings.h"

namespace py = pybind11;

// Declaration of the external function
std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateHomographyGravity(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    const Eigen::Matrix3d &kGravitySource_, // The gravity alignment matrix of the source camera
    const Eigen::Matrix3d &kGravityDestination_, // The gravity alignment matrix of the destination camera
    stereoglue::RANSACSettings &settings_); // The RANSAC settings
    
std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateHomographySimple(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    stereoglue::RANSACSettings &settings_); // The RANSAC settings
    
std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateFundamentalMatrix(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    stereoglue::RANSACSettings &settings_); // The RANSAC settings

std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateEssentialMatrixGravity(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    const Eigen::Matrix3d &kGravitySource_, // The gravity alignment matrix of the source camera
    const Eigen::Matrix3d &kGravityDestination_, // The gravity alignment matrix of the destination camera
    stereoglue::RANSACSettings &settings_); // The RANSAC settings

PYBIND11_MODULE(pystereoglue, m) {
    m.doc() = "Python bindings for the RANSAC C++ library using pybind11";

    // Expose the sampler types to Python
    py::enum_<stereoglue::scoring::ScoringType>(m, "ScoringType")
        .value("RANSAC", stereoglue::scoring::ScoringType::RANSAC)
        .value("MSAC", stereoglue::scoring::ScoringType::MSAC)
        .value("MAGSAC", stereoglue::scoring::ScoringType::MAGSAC)
        .export_values();

    // Expose the sampler types to Python
    py::enum_<stereoglue::samplers::SamplerType>(m, "SamplerType")
        .value("Uniform", stereoglue::samplers::SamplerType::Uniform)
        .export_values();

    // Expose the LO types to Python
    py::enum_<stereoglue::local_optimization::LocalOptimizationType>(m, "LocalOptimizationType")
        .value("Nothing", stereoglue::local_optimization::LocalOptimizationType::None)
        .value("LSQ", stereoglue::local_optimization::LocalOptimizationType::LSQ)
        .value("IteratedLSQ", stereoglue::local_optimization::LocalOptimizationType::IRLS)
        .value("NestedRANSAC", stereoglue::local_optimization::LocalOptimizationType::NestedRANSAC)
        .export_values();

    // Expose the Termination types to Python
    py::enum_<stereoglue::termination::TerminationType>(m, "TerminationType")
        .value("RANSAC", stereoglue::termination::TerminationType::RANSAC)
        .export_values();

    py::class_<stereoglue::LocalOptimizationSettings>(m, "LocalOptimizationSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &stereoglue::LocalOptimizationSettings::maxIterations)
        .def_readwrite("sample_size_multiplier", &stereoglue::LocalOptimizationSettings::sampleSizeMultiplier);

    // Expose the RANSAC settings to Python
    py::class_<stereoglue::RANSACSettings>(m, "RANSACSettings")
        .def(py::init<>())
        .def_readwrite("min_iterations", &stereoglue::RANSACSettings::minIterations)
        .def_readwrite("max_iterations", &stereoglue::RANSACSettings::maxIterations)
        .def_readwrite("inlier_threshold", &stereoglue::RANSACSettings::inlierThreshold)
        .def_readwrite("confidence", &stereoglue::RANSACSettings::confidence)
        .def_readwrite("scoring", &stereoglue::RANSACSettings::scoring)
        .def_readwrite("sampler", &stereoglue::RANSACSettings::sampler)
        .def_readwrite("core_number", &stereoglue::RANSACSettings::coreNumber)
        .def_readwrite("local_optimization", &stereoglue::RANSACSettings::localOptimization)
        .def_readwrite("final_optimization", &stereoglue::RANSACSettings::finalOptimization)
        .def_readwrite("termination_criterion", &stereoglue::RANSACSettings::terminationCriterion)
        .def_readwrite("local_optimization_settings", &stereoglue::RANSACSettings::localOptimizationSettings)
        .def_readwrite("final_optimization_settings", &stereoglue::RANSACSettings::finalOptimizationSettings);
        
    // Expose the function to Python
    m.def("estimateHomographyGravity", &estimateHomographyGravity, "A function that performs homography estimation from point correspondences.",
        py::arg("lafs1"),
        py::arg("lafs2"),
        py::arg("matches"),
        py::arg("scores"),
        py::arg("intrinsics_src"),
        py::arg("intrinsics_dst"),
        py::arg("image_sizes"),
        py::arg("gravity_src") = Eigen::Matrix3d::Identity(),
        py::arg("gravity_dst") = Eigen::Matrix3d::Identity(),
        py::arg("config") = stereoglue::RANSACSettings());
        
    m.def("estimateHomographySimple", &estimateHomographySimple, "A function that performs homography estimation from point correspondences.",
        py::arg("lafs1"),
        py::arg("lafs2"),
        py::arg("matches"),
        py::arg("scores"),
        py::arg("intrinsics_src"),
        py::arg("intrinsics_dst"),
        py::arg("image_sizes"),
        py::arg("config") = stereoglue::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateFundamentalMatrix", &estimateFundamentalMatrix, "A function that performs fundamental matrix estimation from point correspondences.",
        py::arg("lafs1"),
        py::arg("lafs2"),
        py::arg("matches"),
        py::arg("scores"),
        py::arg("image_sizes"),
        py::arg("config") = stereoglue::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateEssentialMatrixGravity", &estimateEssentialMatrixGravity, "A function that performs essential matrix estimation with known gravity.",
        py::arg("lafs1"),
        py::arg("lafs2"),
        py::arg("matches"),
        py::arg("scores"),
        py::arg("intrinsics_src"),
        py::arg("intrinsics_dst"),
        py::arg("image_sizes"),
        py::arg("gravity_src") = Eigen::Matrix3d::Identity(),
        py::arg("gravity_dst") = Eigen::Matrix3d::Identity(),
        py::arg("config") = stereoglue::RANSACSettings());
}
