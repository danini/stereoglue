// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of ETH Zurich nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include "../utils/macros.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../estimators/abstract_estimator.h"
#include "abstract_scoring.h"
#include "score.h"
#include <Eigen/Core>

namespace stereoglue {
namespace scoring {

class RANSACScoring : public AbstractScoring
{
    public:
        // Constructor 
        RANSACScoring() {}

        // Destructor
        ~RANSACScoring() {}

        // Set the threshold
        FORCE_INLINE void setThreshold(const double kThreshold_)
        {
            threshold = kThreshold_;
            squaredThreshold = threshold * threshold;
        }

        // Sample function
        FORCE_INLINE Score score(
            const DataMatrix &kDataSrc_, // Data matrix for the source image
            const DataMatrix &kDataDst_, // Data matrix for the destination image
            const DataMatrix &kMatches_, // Data matrix for the matches
            const DataMatrix &kMatchScores_, // Data matrix for the match scores
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<std::pair<size_t, size_t>> &inliers_, // Inlier indices
            const bool kStoreInliers_ = true, // Store inliers or not
            const Score& kBestScore_ = Score()) const // The potential inlier sets from the inlier selector
        {
            // Create a static empty Score
            static const Score kEmptyScore;
            // The number of points
            const int kPointNumber = kMatches_.rows();
            // The size of the pool
            const int kPoolSize = kMatches_.cols();
            // The squared residual
            double squaredResidual;
            // Score and inlier number
            int inlierNumber = 0;
            // The score of the previous best model
            const double kBestInlierNumber = kBestScore_.getInlierNumber();
            // Inlier pair
            std::pair<size_t, size_t> inlierPair;

            // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
            for (size_t srcIdx = 0; srcIdx < kPointNumber; ++srcIdx)
            {
                for (size_t poolIdx = 0; poolIdx < kPoolSize; ++poolIdx)
                {
                    const int dstIdx = static_cast<int>(kMatches_(srcIdx, poolIdx));

                    // If the match is invalid, skip it
                    // This assumes that the potential matches are ordered. Otherwise, "continue" should be used.
                    if (dstIdx < 0)
                        break;

                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(
                            kDataSrc_.row(srcIdx),
                            kDataDst_.row(dstIdx),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        if (kStoreInliers_) // Store the point as an inlier if needed.
                        {
                            inlierPair.first = srcIdx;
                            inlierPair.second = dstIdx;
                            inliers_.emplace_back(inlierPair);
                        }

                        // Increase the inlier number
                        ++inlierNumber;
                        
                        // Interrupt if the current match is an inlier
                        break;
                    }
                }

                // Interrupt if there is no chance of being better than the best model
                if (kPointNumber - srcIdx + inlierNumber < kBestInlierNumber)
                    return kEmptyScore;
            }
            
            return Score(inlierNumber, inlierNumber);
        }

        // Sample function
        FORCE_INLINE Score score(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<size_t> &inliers_, // Inlier indices
            const bool kStoreInliers_ = true,
            const Score& kBestScore_ = Score(),
            std::vector<const std::vector<size_t>*> *kPotentialInlierSets_ = nullptr) const // The potential inlier sets from the inlier selector
        {   
            // Create a static empty Score
            static const Score kEmptyScore;
            // The number of points
            const int kPointNumber = kData_.rows();
            // The squared residual
            double squaredResidual;
            // Score and inlier number
            int inlierNumber = 0;
            // The score of the previous best model
            const double kBestInlierNumber = kBestScore_.getInlierNumber();

            // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
            if (kPotentialInlierSets_ != nullptr)
            {
                size_t testedPointNumber = 0;
                for (const auto &potentialInlierSet : *kPotentialInlierSets_)
                {
                    for (const auto &pointIdx : *potentialInlierSet)
                    {
                        // Calculate the point-to-model residual
                        squaredResidual =
                            kEstimator_->squaredResidual(kData_.row(pointIdx),
                                kModel_);

                        // If the residual is smaller than the threshold, store it as an inlier and
                        // increase the score.
                        if (squaredResidual < squaredThreshold)
                        {
                            if (kStoreInliers_) // Store the point as an inlier if needed.
                                inliers_.emplace_back(pointIdx);

                            // Increase the inlier number
                            ++inlierNumber;
                        }

                        // Interrupt if there is no chance of being better than the best model
                        if (kPointNumber - testedPointNumber + inlierNumber < kBestInlierNumber)
                            return kEmptyScore;
                        ++testedPointNumber;
                    }
                }
            } else
            {
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row(pointIdx),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        if (kStoreInliers_) // Store the point as an inlier if needed.
                            inliers_.emplace_back(pointIdx);

                        // Increase the inlier number
                        ++inlierNumber;
                    }

                    // Interrupt if there is no chance of being better than the best model
                    if (kPointNumber - pointIdx + inlierNumber < kBestInlierNumber)
                        return kEmptyScore;
                }
            }

            return Score(inlierNumber, inlierNumber);
        }

        // Get weights for the points
        FORCE_INLINE void getWeights(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<double> &weights_, // The weights of the points
            const std::vector<size_t> *kIndices_ = nullptr) const  // The indices of the points
        {
            if (kIndices_ == nullptr)
            {
                // The number of points
                const int kPointNumber = kData_.rows();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row(pointIdx),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                        weights_[pointIdx] = 1.0;
                    else
                        weights_[pointIdx] = 0.0;
                }
            }
            else
            {
                // The number of points
                const int kPointNumber = kIndices_->size();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row((*kIndices_)[pointIdx]),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                        weights_[pointIdx] = 1.0;
                    else
                        weights_[pointIdx] = 0.0;
                }
            }
        }
};

}
}