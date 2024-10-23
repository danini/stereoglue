#include <Eigen/Dense>

#include "stereoglue.h"

#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_homography_one_affine_gravity.h"
#include "estimators/solver_homography_one_affine_approximate.h"
#include "estimators/estimator_homography.h"

#include "estimators/solver_essential_matrix_five_point_nister.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_focal_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_essential_matrix_one_affine_gravity.h"
#include "estimators/estimator_essential_matrix.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_fundamental_matrix_bundle_adjustment.h"
#include "estimators/solver_focal_fundamental_matrix_bundle_adjustment.h"
#include "estimators/solver_fundamental_matrix_single_affine_monodepth.h"
#include "estimators/estimator_focal_fundamental_matrix.h"
#include "estimators/estimator_fundamental_matrix.h"
#include "estimators/numerical_optimizer/types.h"

#include "stereoglue.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "utils/types.h"
#include "utils/utils_point_correspondence.h"

#include "correspondence_factory.h"

void initializeLocalOptimizer(
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> &localOptimizer_,
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimizationType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const stereoglue::RANSACSettings &kSettings_, // The RANSAC settings
    const stereoglue::LocalOptimizationSettings &kLOSettings_, // The RANSAC settings
    const stereoglue::models::Types &kModelType_,
    const bool kFinalOptimization_ = false) 
{
    if (kLocalOptimizationType_ == stereoglue::local_optimization::LocalOptimizationType::None)
        return;

    if (kLocalOptimizationType_ == stereoglue::local_optimization::LocalOptimizationType::IRLS)
    {
        // Set the neighborhood graph to the local optimizer
        auto irlsLocalOptimizer = dynamic_cast<stereoglue::local_optimization::IRLSOptimizer *>(localOptimizer_.get());
        irlsLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);        
    }  else if (kLocalOptimizationType_ == stereoglue::local_optimization::LocalOptimizationType::LSQ)
    {
        // Set the neighborhood graph to the local optimizer
        auto lsqLocalOptimizer = dynamic_cast<stereoglue::local_optimization::LeastSquaresOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ || kModelType_ == stereoglue::models::Types::Homography)
            lsqLocalOptimizer->setUseInliers(true);
    }  else if (kLocalOptimizationType_ == stereoglue::local_optimization::LocalOptimizationType::NestedRANSAC)
    {
        // Set the neighborhood graph to the local optimizer
        auto nestedRansacLocalOptimizer = dynamic_cast<stereoglue::local_optimization::NestedRANSACOptimizer *>(localOptimizer_.get());
        nestedRansacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations); 
        nestedRansacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier); 
    }
}

std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateFundamentalMatrix(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    stereoglue::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kMatches_.rows() < 1) 
        throw std::invalid_argument("The input matrix must have at least 1 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (kMatches_.rows() != kMatchScores_.rows())
        throw std::invalid_argument("The probabilities must have the same number of elements as the number of LAFs.");
    if (kLafsSrc_.cols() != 9 || kLafsDst_.cols() != 9)
        throw std::invalid_argument("The input matrix must have 9 columns (x1, y1, x2, y2, a11, a12, a21, a22, lambda, dx, dy).");

    // Normalize the point correspondences
    DataMatrix normalizedLafsSrc, normalizedLafsDst;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    
    double scale = 1.0;
    if (kMatches_.rows() >= 3)
    {
        normalizeLAFs(
            kLafsSrc_,
            normalizingTransformSource,
            normalizedLafsSrc);
            
        normalizeLAFs(
            kLafsDst_,
            normalizingTransformDestination,
            normalizedLafsDst);

        scale = 0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
        settings_.inlierThreshold *= scale;
    } else
    {
        normalizedLafsSrc = kLafsSrc_;
        normalizedLafsDst = kLafsDst_;
        normalizingTransformSource = Eigen::Matrix3d::Identity();
        normalizingTransformDestination = Eigen::Matrix3d::Identity();
    }
        
    // Get the values from the settings
    const stereoglue::scoring::ScoringType kScoring = settings_.scoring;
    const stereoglue::samplers::SamplerType kSampler = settings_.sampler;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::FocalFundamentalMatrixEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::FocalFundamentalMatrixEstimator>(new stereoglue::estimator::FocalFundamentalMatrixEstimator());
    estimator->setMinimalSolver(new stereoglue::estimator::solver::FundamentalMatrixSingleAffineDepthSolver());
    estimator->setNonMinimalSolver(new stereoglue::estimator::solver::FocalEssentialMatrixBundleAdjustmentSolver());
    auto &solverOptions = dynamic_cast<stereoglue::estimator::solver::FocalEssentialMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the scoring object
    std::unique_ptr<stereoglue::scoring::AbstractScoring> scorer = 
        stereoglue::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == stereoglue::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<stereoglue::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        // solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<stereoglue::termination::AbstractCriterion> terminationCriterion = 
        stereoglue::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == stereoglue::termination::TerminationType::RANSAC)
        dynamic_cast<stereoglue::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the correspondence factory that will compose correspondences from the matches
    std::unique_ptr<stereoglue::AffineDepthCorrespondenceFactory> correspondenceFactory = 
        std::make_unique<stereoglue::AffineDepthCorrespondenceFactory>();

    // Create the RANSAC object
    stereoglue::StereoGlue stereoglue;
    stereoglue.setEstimator(estimator.get()); // Set the estimator
    stereoglue.setScoring(scorer.get()); // Set the scoring method
    stereoglue.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion
    stereoglue.setCorrespondenceFactory(correspondenceFactory.get()); // Set the correspondence factory

    // Set the local optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            localOptimizer,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            stereoglue::models::Types::FundamentalMatrix);
            
        // Set the local optimizer
        stereoglue.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kFinalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            finalOptimizer,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            stereoglue::models::Types::FundamentalMatrix);
            
        // Set the final optimizer
        stereoglue.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    stereoglue.setSettings(settings_);
    
    // Run the robust estimator
    stereoglue.run(normalizedLafsSrc,
        normalizedLafsDst,
        kMatches_,
        kMatchScores_);

    // Check if the model is valid
    if (stereoglue.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, stereoglue.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d fundamentalMatrix = stereoglue.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Return the best model with the inliers and the score
    return std::make_tuple(fundamentalMatrix, 
        stereoglue.getInliers(), 
        stereoglue.getBestScore().getValue(), 
        stereoglue.getIterationNumber());
}

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
    stereoglue::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kMatches_.rows() < 6) 
        throw std::invalid_argument("The input matrix must have at least 6 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (kMatches_.rows() != kMatchScores_.rows())
        throw std::invalid_argument("The probabilities must have the same number of elements as the number of LAFs.");

    // Normalize the point correspondences
    DataMatrix normalizedLafsSrc, normalizedLafsDst;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    
    normalizeLAFsByIntrinsics(
        kLafsSrc_,
        kIntrinsicsSource_,
        normalizedLafsSrc);

    normalizeLAFsByIntrinsics(
        kLafsDst_,
        kIntrinsicsDestination_,
        normalizedLafsDst);

    const double kScale = 
        0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));
    settings_.inlierThreshold /= kScale;

    // Get the values from the settings
    const stereoglue::scoring::ScoringType kScoring = settings_.scoring;
    const stereoglue::samplers::SamplerType kSampler = settings_.sampler;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::EssentialMatrixEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::EssentialMatrixEstimator>(new stereoglue::estimator::EssentialMatrixEstimator());
    estimator->setMinimalSolver(new stereoglue::estimator::solver::EssentialMatrixOneAffineGravity());
    estimator->setNonMinimalSolver(new stereoglue::estimator::solver::EssentialMatrixBundleAdjustmentSolver());
    auto &solverOptions = dynamic_cast<stereoglue::estimator::solver::EssentialMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Set the gravity to the minimal solver
    dynamic_cast<stereoglue::estimator::solver::EssentialMatrixOneAffineGravity *>(estimator->getMutableMinimalSolver())->setGravity(
        kGravitySource_, // The gravity alignment matrix of the source camera
        kGravityDestination_); // The gravity alignment matrix of the destination camera

    // Create the scoring object
    std::unique_ptr<stereoglue::scoring::AbstractScoring> scorer = 
        stereoglue::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == stereoglue::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<stereoglue::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        //solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<stereoglue::termination::AbstractCriterion> terminationCriterion = 
        stereoglue::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == stereoglue::termination::TerminationType::RANSAC)
        dynamic_cast<stereoglue::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the correspondence factory that will compose correspondences from the matches
    std::unique_ptr<stereoglue::AbstractCorrespondenceFactory> correspondenceFactory = 
        std::make_unique<stereoglue::AffineCorrespondenceFactory>();

    // Create the RANSAC object
    stereoglue::StereoGlue stereoglue;
    stereoglue.setEstimator(estimator.get()); // Set the estimator
    stereoglue.setScoring(scorer.get()); // Set the scoring method
    stereoglue.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion
    stereoglue.setCorrespondenceFactory(correspondenceFactory.get()); // Set the correspondence factory

    // Set the local optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            localOptimizer,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            stereoglue::models::Types::EssentialMatrix);
            
        // Set the local optimizer
        stereoglue.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kFinalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            finalOptimizer,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            stereoglue::models::Types::EssentialMatrix);
            
        // Set the final optimizer
        stereoglue.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    stereoglue.setSettings(settings_);
    
    // Run the robust estimator
    stereoglue.run(normalizedLafsSrc,
        normalizedLafsDst,
        kMatches_,
        kMatchScores_);

    // Check if the model is valid
    if (stereoglue.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, stereoglue.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d essentialMatrix = stereoglue.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(essentialMatrix, 
        stereoglue.getInliers(), 
        stereoglue.getBestScore().getValue(), 
        stereoglue.getIterationNumber());
}

// Declaration of the external function
std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateHomographySimple(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    stereoglue::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kMatches_.rows() < 4) 
        throw std::invalid_argument("The input matrix must have at least 4 rows to estimate the homography accurately.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (kMatches_.rows() != kMatchScores_.rows())
        throw std::invalid_argument("The probabilities must have the same number of elements as the number of LAFs.");

    // Normalize the point correspondences
    DataMatrix normalizedLafsSrc, normalizedLafsDst;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    
    normalizeLAFsByIntrinsics(
        kLafsSrc_,
        kIntrinsicsSource_,
        normalizedLafsSrc);

    normalizeLAFsByIntrinsics(
        kLafsDst_,
        kIntrinsicsDestination_,
        normalizedLafsDst);

    const double kScale = 
        0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));
    settings_.inlierThreshold /= kScale;

    // Get the values from the settings
    const stereoglue::scoring::ScoringType kScoring = settings_.scoring;
    const stereoglue::samplers::SamplerType kSampler = settings_.sampler;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::HomographyEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::HomographyEstimator>(new stereoglue::estimator::HomographyEstimator());
    estimator->setMinimalSolver(new stereoglue::estimator::solver::HomographyOneAffineApproximateSolver());
    estimator->setNonMinimalSolver(new stereoglue::estimator::solver::HomographyFourPointSolver());

    // Create the scoring object
    std::unique_ptr<stereoglue::scoring::AbstractScoring> scorer = 
        stereoglue::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == stereoglue::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
        dynamic_cast<stereoglue::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());

    // Create termination criterion object
    std::unique_ptr<stereoglue::termination::AbstractCriterion> terminationCriterion = 
        stereoglue::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == stereoglue::termination::TerminationType::RANSAC)
        dynamic_cast<stereoglue::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the correspondence factory that will compose correspondences from the matches
    std::unique_ptr<stereoglue::AbstractCorrespondenceFactory> correspondenceFactory = 
        std::make_unique<stereoglue::AffineCorrespondenceFactory>();

    // Create the RANSAC object
    stereoglue::StereoGlue stereoglue;
    stereoglue.setEstimator(estimator.get()); // Set the estimator
    stereoglue.setScoring(scorer.get()); // Set the scoring method
    stereoglue.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion
    stereoglue.setCorrespondenceFactory(correspondenceFactory.get()); // Set the correspondence factory

    // Set the local optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            localOptimizer,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            stereoglue::models::Types::Homography);
            
        // Set the local optimizer
        stereoglue.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kFinalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            finalOptimizer,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            stereoglue::models::Types::Homography);
            
        // Set the final optimizer
        stereoglue.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    stereoglue.setSettings(settings_);
    
    // Run the robust estimator
    stereoglue.run(normalizedLafsSrc,
        normalizedLafsDst,
        kMatches_,
        kMatchScores_);

    // Check if the model is valid
    if (stereoglue.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, stereoglue.getIterationNumber());

    // Get the normalized homography
    Eigen::Matrix3d homography = stereoglue.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(homography, 
        stereoglue.getInliers(), 
        stereoglue.getBestScore().getValue(), 
        stereoglue.getIterationNumber());
    return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, 0);
}

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
    stereoglue::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kMatches_.rows() < 4) 
        throw std::invalid_argument("The input matrix must have at least 4 rows to estimate the homography accurately.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (kMatches_.rows() != kMatchScores_.rows())
        throw std::invalid_argument("The probabilities must have the same number of elements as the number of LAFs.");

    // Normalize the point correspondences
    DataMatrix normalizedLafsSrc, normalizedLafsDst;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    
    normalizeLAFsByIntrinsics(
        kLafsSrc_,
        kIntrinsicsSource_,
        normalizedLafsSrc);

    normalizeLAFsByIntrinsics(
        kLafsDst_,
        kIntrinsicsDestination_,
        normalizedLafsDst);

    const double kScale = 
        0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));
    settings_.inlierThreshold /= kScale;

    // Get the values from the settings
    const stereoglue::scoring::ScoringType kScoring = settings_.scoring;
    const stereoglue::samplers::SamplerType kSampler = settings_.sampler;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::HomographyEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::HomographyEstimator>(new stereoglue::estimator::HomographyEstimator());
    estimator->setMinimalSolver(new stereoglue::estimator::solver::HomographyOneAffineGravitySolver());
    estimator->setNonMinimalSolver(new stereoglue::estimator::solver::HomographyFourPointSolver());

    // Set the gravity to the minimal solver
    dynamic_cast<stereoglue::estimator::solver::HomographyOneAffineGravitySolver *>(estimator->getMutableMinimalSolver())->setGravity(
        kGravitySource_, // The gravity alignment matrix of the source camera
        kGravityDestination_); // The gravity alignment matrix of the destination camera

    // Create the scoring object
    std::unique_ptr<stereoglue::scoring::AbstractScoring> scorer = 
        stereoglue::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == stereoglue::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
        dynamic_cast<stereoglue::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());

    // Create termination criterion object
    std::unique_ptr<stereoglue::termination::AbstractCriterion> terminationCriterion = 
        stereoglue::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == stereoglue::termination::TerminationType::RANSAC)
        dynamic_cast<stereoglue::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the correspondence factory that will compose correspondences from the matches
    std::unique_ptr<stereoglue::AbstractCorrespondenceFactory> correspondenceFactory = 
        std::make_unique<stereoglue::AffineCorrespondenceFactory>();

    // Create the RANSAC object
    stereoglue::StereoGlue stereoglue;
    stereoglue.setEstimator(estimator.get()); // Set the estimator
    stereoglue.setScoring(scorer.get()); // Set the scoring method
    stereoglue.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion
    stereoglue.setCorrespondenceFactory(correspondenceFactory.get()); // Set the correspondence factory

    // Set the local optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            localOptimizer,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            stereoglue::models::Types::Homography);
            
        // Set the local optimizer
        stereoglue.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kFinalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            finalOptimizer,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            stereoglue::models::Types::Homography);
            
        // Set the final optimizer
        stereoglue.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    stereoglue.setSettings(settings_);
    
    // Run the robust estimator
    stereoglue.run(normalizedLafsSrc,
        normalizedLafsDst,
        kMatches_,
        kMatchScores_);

    // Check if the model is valid
    if (stereoglue.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, stereoglue.getIterationNumber());

    // Get the normalized homography
    Eigen::Matrix3d homography = stereoglue.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(homography, 
        stereoglue.getInliers(), 
        stereoglue.getBestScore().getValue(), 
        stereoglue.getIterationNumber());
    return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, 0);
}