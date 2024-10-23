#include "stereoglue.h"

#include <iostream>

namespace stereoglue {

StereoGlue::StereoGlue() : 
    currentSample(nullptr), 
    localOptimizer(nullptr), 
    finalOptimizer(nullptr)
{

}

StereoGlue::~StereoGlue() 
{
    
}

void StereoGlue::run(
    const DataMatrix &kDataSrc_,
    const DataMatrix &kDataDst_,
    const DataMatrix &kMatches_,
    const DataMatrix &kMatchScores_)
{
    // Initialize the variables
    const double &kThreshold_ = scoring->getThreshold();
    bool isModelUpdated,
        immediateTermination = false;
    const size_t kStrickIterationLimit = settings.maxIterations;
    maxIterations = settings.maxIterations;

    bestScore = scoring::Score(); // The best score
    std::vector<scoring::Score> bestScores(settings.coreNumber);
    std::vector<models::Model> bestModels(settings.coreNumber);
    std::vector<std::vector<std::pair<size_t, size_t>>> bestInliers(settings.coreNumber);
    std::vector<std::vector<std::pair<size_t, size_t>>> tmpInlierSets(settings.coreNumber);
    
    // Iterate through the matches
#pragma omp parallel for num_threads(settings.coreNumber)
    for (size_t coreIdx = 0; coreIdx < settings.coreNumber; ++coreIdx)
    {
        // Initialize the variables
        size_t fromIdx = coreIdx * kMatches_.rows() / settings.coreNumber; // The starting index
        size_t toIdx = (coreIdx < settings.coreNumber - 1) ?
            (coreIdx + 1) * kMatches_.rows() / settings.coreNumber :
            kMatches_.rows(); // The ending index

        size_t iterationNumber = 0; // The number of iterations

        auto &tmpInliers = tmpInlierSets[coreIdx]; // The temporary inliers
        tmpInliers.reserve(kDataSrc_.rows()); // Reserve memory for the temporary inliers

        auto &inliers = bestInliers[coreIdx]; // The current best inliers
        inliers.reserve(kDataSrc_.rows()); // Reserve memory for the inliers

        size_t *currentSample = new size_t[1]; // The current sample
        currentSample[0] = 0; // The index of the sample is always 0.

        auto &bestModel = bestModels[coreIdx]; // The best model for this code

        auto &bestScore = bestScores[coreIdx]; // The best score
        bestScore = scoring::Score(); // Initialize the best score

        std::vector<models::Model> currentModels; // The current models estimated from the current sample
        scoring::Score currentScore; // The score of the current model
        models::Model locallyOptimizedModel; // The locally optimized model
        Eigen::MatrixXd correspondence(1, correspondenceFactory->dimensions()); // The correspondence to be created in every iteration    

        // Iterate through the matches
        for (size_t srcIdx = fromIdx; srcIdx < toIdx; ++srcIdx)
        {
            // Get the source and destination indices
            for (size_t poolIdx = 0; poolIdx < kMatches_.cols(); ++poolIdx)
            {
                // Check if the match is valid
                if (kMatches_(srcIdx, poolIdx) < 0)
                    continue;

                // Increase the iteration number
                ++iterationNumber;

                // Get the destination index
                size_t dstIdx =
                    static_cast<size_t>(kMatches_(srcIdx, poolIdx));

                // Create the correspondence
                correspondenceFactory->create(
                    kDataSrc_, // The source data
                    kDataDst_, // The destination data
                    srcIdx, // The source index
                    dstIdx, // The destination index
                    correspondence); // The correspondence

                // Estimate the model from the current correspondence
                currentModels.clear(); // Clearing the current mode
                if (!estimator->estimateModel(correspondence, // The current correspondence
                    currentSample, // Selected minimal sample
                    &currentModels)) // Estimated models
                {
                    continue;
                }

                // Iterate through the models
                bool isModelUpdated = false;
                for (models::Model &model : currentModels)
                {
                    // Clear the inliers
                    tmpInliers.clear();

                    // Score the model
                    currentScore = scoring->score(
                        kDataSrc_, // The source data
                        kDataDst_, // The destination data
                        kMatches_, // The matches
                        kMatchScores_, // The match scores
                        model, // The model to be scored
                        estimator, // Estimator
                        tmpInliers, // Inlier indices
                        true, // Store inliers
                        bestScore); // The best score

                    if (bestScore < currentScore)
                    {
                        // Update the best model
                        bestScore = currentScore;
                        bestModel = model;
                        inliers.swap(tmpInliers);
                        isModelUpdated = true;
                    }
                }
                
                if (isModelUpdated)
                {
                    // Perform local optimization if needed
                    if (localOptimizer != nullptr)
                    {
                        tmpInliers.clear();
                        localOptimizer->run(
                            kDataSrc_, // The source data
                            kDataDst_, // The destination data
                            kMatches_, // The matches
                            kMatchScores_, // The match scores
                            inliers, // Inliers
                            bestModel, // The best model
                            bestScore, // The score of the best model
                            estimator, // Estimator
                            scoring, // Scoring object
                            locallyOptimizedModel, // The locally optimized model
                            currentScore, // The score of the current model
                            tmpInliers); // The inliers of the estimated model

                        if (bestScore < currentScore)
                        {
                            // Update the best model
                            bestScore = currentScore;
                            bestModel = locallyOptimizedModel;
                            inliers.swap(tmpInliers);
                        } 
                    }
                }

                if (iterationNumber >= maxIterations / settings.coreNumber)
                    break;
            }

            if (iterationNumber >= maxIterations / settings.coreNumber)
                break;
        }

        // Clean up
        delete[] currentSample;
    }

    // Select the best model from the cores
    for (size_t coreIdx = 0; coreIdx < settings.coreNumber; ++coreIdx)
    {
        if (bestScores[coreIdx] > bestScore)
        {
            bestScore = bestScores[coreIdx];
            bestModel = bestModels[coreIdx];
            inliers.swap(bestInliers[coreIdx]);
        }
    }

    // Perform final optimization if needed
    if (finalOptimizer != nullptr && inliers.size() > 1)
    {
        scoring::Score currentScore; // The score of the current model
        models::Model locallyOptimizedModel; // The locally optimized model
        
        tmpInlierSets[0].clear();
        finalOptimizer->run(
            kDataSrc_, // The source data
            kDataDst_, // The destination data
            kMatches_, // The matches
            kMatchScores_, // The match scores
            inliers, // Inliers
            bestModel, // The best model
            bestScore, // The score of the best model
            estimator, // Estimator
            scoring, // Scoring object
            locallyOptimizedModel, // The locally optimized model
            currentScore, // The score of the current model
            tmpInlierSets[0]); // The inliers of the estimated model

        // Update the best model
        if (currentScore.getValue() > bestScore.getValue())
        {
            bestScore = currentScore;
            bestModel = locallyOptimizedModel;
            inliers.swap(tmpInlierSets[0]);
        }
    }
}

/*
    Setters and getters
*/
// Set the scoring object
void StereoGlue::setScoring(scoring::AbstractScoring *scoring_)
{
    scoring = scoring_;
}

// Return a constant pointer to the scoring object
const scoring::AbstractScoring *StereoGlue::getScoring() const
{
    return scoring;
}

// Return a mutable pointer to the scoring object
scoring::AbstractScoring *StereoGlue::getMutableScoring()
{
    return scoring;
}

// Set the sampler
void StereoGlue::setSampler(samplers::AbstractSampler *sampler_)
{
    sampler = sampler_;
}

// Return a constant pointer to the sampler
const samplers::AbstractSampler *StereoGlue::getSampler() const
{
    return sampler;
}

// Return a mutable pointer to the sampler
samplers::AbstractSampler *StereoGlue::getMutableSampler()
{
    return sampler;
}

// Set the settings
void StereoGlue::setSettings(const RANSACSettings &kSettings_)
{
    settings = kSettings_;
}

// Return the settings
const RANSACSettings &StereoGlue::getSettings() const
{
    return settings;
}

// Return a mutable reference to the settings
RANSACSettings &StereoGlue::getMutableSettings()
{
    return settings;
}

// Set the estimator
void StereoGlue::setEstimator(estimator::Estimator *estimator_)
{
    estimator = estimator_;
}

// Return a constant pointer to the estimator
const estimator::Estimator *StereoGlue::getEstimator() const
{
    return estimator;
}

// Return a mutable pointer to the estimator
estimator::Estimator *StereoGlue::getMutableEstimator()
{
    return estimator;
}

// Get the best model
const models::Model &StereoGlue::getBestModel() const
{
    return bestModel;
}

// Get the inliers of the best model
const std::vector<std::pair<size_t, size_t>> &StereoGlue::getInliers() const
{
    return inliers;
}

// Get the score of the best model
const scoring::Score &StereoGlue::getBestScore() const
{
    return bestScore;
}

// Get the number of iterations
size_t StereoGlue::getIterationNumber() const
{
    return iterationNumber;
}

// Set the local optimization object
void StereoGlue::setLocalOptimizer(local_optimization::LocalOptimizer *localOptimizer_)
{
    localOptimizer = localOptimizer_;
}

// Return a constant pointer to the local optimization object
const local_optimization::LocalOptimizer *StereoGlue::getLocalOptimizer() const
{
    return localOptimizer;
}

// Return a mutable pointer to the local optimization object
local_optimization::LocalOptimizer *StereoGlue::getMutableLocalOptimizer()
{
    return localOptimizer;
}

// Set the local optimization object
void StereoGlue::setFinalOptimizer(local_optimization::LocalOptimizer *finalOptimizer_)
{
    finalOptimizer = finalOptimizer_;
}

// Return a constant pointer to the local optimization object
const local_optimization::LocalOptimizer *StereoGlue::getFinalOptimizer() const
{
    return finalOptimizer;
}

// Return a mutable pointer to the local optimization object
local_optimization::LocalOptimizer *StereoGlue::getMutableFinalOptimizer()
{
    return finalOptimizer;
}

// Set the termination criterion object
void StereoGlue::setTerminationCriterion(termination::AbstractCriterion *terminationCriterion_)
{
    terminationCriterion = terminationCriterion_;
}

// Return a constant pointer to the termination criterion object
const termination::AbstractCriterion *StereoGlue::getTerminationCriterion() const
{
    return terminationCriterion;
}

// Return a mutable pointer to the termination criterion object
termination::AbstractCriterion *StereoGlue::getMutableTerminationCriterion()
{
    return terminationCriterion;
}

// Set the scoring object
void StereoGlue::setCorrespondenceFactory(AbstractCorrespondenceFactory *correspondenceFactory_)
{
    correspondenceFactory = correspondenceFactory_;
}

// Return a constant pointer to the scoring object
const AbstractCorrespondenceFactory *StereoGlue::getCorrespondenceFactory() const
{
    return correspondenceFactory;
}

// Return a mutable pointer to the scoring object
AbstractCorrespondenceFactory *StereoGlue::getMutableCorrespondenceFactory()
{
    return correspondenceFactory;
}
}