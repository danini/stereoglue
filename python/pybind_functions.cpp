#include <Eigen/Dense>

#include "stereoglue.h"

#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_homography_one_affine_gravity.h"
#include "estimators/estimator_homography.h"

#include "estimators/solver_essential_matrix_five_point_nister.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_essential_matrix_one_affine_gravity.h"
#include "estimators/estimator_essential_matrix.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_fundamental_matrix_bundle_adjustment.h"
#include "estimators/estimator_fundamental_matrix.h"
#include "estimators/numerical_optimizer/types.h"

#include "stereoglue.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "neighborhood/types.h"
#include "inlier_selectors/types.h"
#include "utils/types.h"
#include "utils/utils_point_correspondence.h"

#include "correspondence_factory.h"

// Function to initialize the neighborhood graph
template <size_t _DimensionNumber>
void initializeNeighborhood(
    const DataMatrix& kCorrespondences_, // The point correspondences
    std::unique_ptr<stereoglue::neighborhood::AbstractNeighborhoodGraph> &neighborhoodGraph_, // The neighborhood graph
    const stereoglue::neighborhood::NeighborhoodType kNeighborhoodType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const stereoglue::RANSACSettings &kSettings_) // The RANSAC settings
{
    // Create the neighborhood graph
    neighborhoodGraph_ = stereoglue::neighborhood::createNeighborhoodGraph<_DimensionNumber>(kNeighborhoodType_);
    // Initialize the neighborhood graph if the neighborhood is grid
    if (kNeighborhoodType_ == stereoglue::neighborhood::NeighborhoodType::Grid) 
    {
        // Check if the image sizes have the correct number of elements
        if (kImageSizes_.size() != _DimensionNumber)
            throw std::invalid_argument("The image sizes must have " + std::to_string(_DimensionNumber) + " elements.");

        // Cast the neighborhood graph to the grid neighborhood graph
        auto gridNeighborhoodGraph = 
            dynamic_cast<stereoglue::neighborhood::GridNeighborhoodGraph<_DimensionNumber> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        const auto &kCellNumber = kSettings_.neighborhoodSettings.neighborhoodGridDensity;
        std::vector<double> kCellSizes(_DimensionNumber);
        for (size_t i = 0; i < _DimensionNumber; i++)
        {
            kCellSizes[i] = kImageSizes_[i] / kCellNumber;
            if (kCellSizes[i] < 1.0)
                throw std::invalid_argument("The cell size is too small (< 1px). Try setting a smaller neighborhood size (in grid it acts as the number of cells along an axis).");
        }

        gridNeighborhoodGraph->setCellSizes(
            kCellSizes, // The sizes of the cells in each dimension
            kCellNumber); // The number of cells in each dimension
    } else if (kNeighborhoodType_ == stereoglue::neighborhood::NeighborhoodType::FLANN_KNN)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<stereoglue::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 0> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setNearestNeighborNumber(kSettings_.neighborhoodSettings.nearestNeighborNumber); 

    } else if (kNeighborhoodType_ == stereoglue::neighborhood::NeighborhoodType::FLANN_Radius)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<stereoglue::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 1> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setRadius(kSettings_.neighborhoodSettings.neighborhoodSize); 

    }
    neighborhoodGraph_->initialize(&kCorrespondences_);
}

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
        if (kFinalOptimization_)
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
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    stereoglue::RANSACSettings &settings_) // The RANSAC settings
{
    return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<std::pair<size_t, size_t>>(), 0.0, 0);
    // Check if the input matrix has the correct dimensions
    /*if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 8) 
        throw std::invalid_argument("The input matrix must have at least 8 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == stereoglue::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalize2D2DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;

    // Get the values from the settings
    const stereoglue::scoring::ScoringType kScoring = settings_.scoring;
    const stereoglue::samplers::SamplerType kSampler = settings_.sampler;
    const stereoglue::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::FundamentalMatrixEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::FundamentalMatrixEstimator>(new stereoglue::estimator::FundamentalMatrixEstimator());
    estimator->setMinimalSolver(new stereoglue::estimator::solver::FundamentalMatrixSevenPointSolver());
    estimator->setNonMinimalSolver(new stereoglue::estimator::solver::FundamentalMatrixBundleAdjustmentSolver());
    auto &solverOptions = dynamic_cast<stereoglue::estimator::solver::FundamentalMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<stereoglue::samplers::AbstractSampler> sampler = 
        stereoglue::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<stereoglue::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == stereoglue::samplers::SamplerType::PROSAC)
        dynamic_cast<stereoglue::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == stereoglue::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<stereoglue::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == stereoglue::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<stereoglue::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == stereoglue::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<stereoglue::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == stereoglue::samplers::SamplerType::ARSampler)
        dynamic_cast<stereoglue::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<stereoglue::scoring::AbstractScoring> scorer = 
        stereoglue::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == stereoglue::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == stereoglue::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        stereoglue::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != stereoglue::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<stereoglue::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == stereoglue::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<stereoglue::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<stereoglue::termination::AbstractCriterion> terminationCriterion = 
        stereoglue::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == stereoglue::termination::TerminationType::RANSAC)
        dynamic_cast<stereoglue::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != stereoglue::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<stereoglue::inlier_selector::AbstractInlierSelector> inlierSelector = 
            stereoglue::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    stereoglue::StereoGlue robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Create the space partitioning inlier selector object
    std::unique_ptr<stereoglue::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != stereoglue::inlier_selector::InlierSelectorType::None)
    { 
        // Check whether the scoring is RANSAC, MSAC, or MAGSAC++
        if (kScoring != stereoglue::scoring::ScoringType::RANSAC &&
            kScoring != stereoglue::scoring::ScoringType::MSAC &&
            kScoring != stereoglue::scoring::ScoringType::MAGSAC)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with RANSAC, MSAC, or MAGSAC++ scoring.");

        // Check if the neighborhood is grid
        if (kNeighborhood != stereoglue::neighborhood::NeighborhoodType::Grid)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with grid neighborhood.");

        // Initialize the neighborhood graph if needed
        if (neighborhoodGraph == nullptr)
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
                
        // Create the inlier selector
        inlierSelector = 
            stereoglue::inlier_selector::createInlierSelector(stereoglue::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC);
        // Initialize the inlier selector
        stereoglue::inlier_selector::SpacePartitioningRANSAC *spacePartitioningRANSAC = 
            reinterpret_cast<stereoglue::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        spacePartitioningRANSAC->initialize(
            neighborhoodGraph.get(), 
            stereoglue::models::Types::FundamentalMatrix);
        spacePartitioningRANSAC->setNormalizers(
            normalizingTransformSource(0, 0), normalizingTransformSource(0, 2), normalizingTransformSource(1, 2),
            normalizingTransformDestination(0, 0), normalizingTransformDestination(0, 2), normalizingTransformDestination(1, 2));
        // Set the inlier selector to the robust estimator
        robustEstimator.setInlierSelector(inlierSelector.get());
    }

    // Set the local optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            stereoglue::models::Types::FundamentalMatrix,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<stereoglue::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != stereoglue::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            stereoglue::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            stereoglue::models::Types::FundamentalMatrix,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d fundamentalMatrix = robustEstimator.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Return the best model with the inliers and the score
    return std::make_tuple(fundamentalMatrix, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());*/
}

std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateEssentialMatrixGravity(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const Eigen::Matrix3d &kGravitySource_, // The gravity alignment matrix of the source camera
    const Eigen::Matrix3d &kGravityDestination_, // The gravity alignment matrix of the destination camera
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
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
    const stereoglue::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
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
std::tuple<Eigen::Matrix3d, std::vector<std::pair<size_t, size_t>>, double, size_t> estimateHomographyGravity(
    const Eigen::MatrixXd& kLafsSrc_, // The local affine frames in the source image
    const Eigen::MatrixXd& kLafsDst_, // The local affine frames in the destination image
    const Eigen::MatrixXd& kMatches_, // The match pool for each point in the source image
    const Eigen::MatrixXd& kMatchScores_, // The match scores for each point in the source image
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const Eigen::Matrix3d &kGravitySource_, // The gravity alignment matrix of the source camera
    const Eigen::Matrix3d &kGravityDestination_, // The gravity alignment matrix of the destination camera
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
    const stereoglue::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const stereoglue::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const stereoglue::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const stereoglue::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<stereoglue::estimator::EssentialMatrixEstimator> estimator = 
        std::unique_ptr<stereoglue::estimator::EssentialMatrixEstimator>(new stereoglue::estimator::HomographyEstimator());
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

    // Get the normalized homography
    Eigen::Matrix3d homography = stereoglue.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(homography, 
        stereoglue.getInliers(), 
        stereoglue.getBestScore().getValue(), 
        stereoglue.getIterationNumber());
}