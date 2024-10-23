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
//     * Neither the name of Czech Technical University nor the
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
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <Eigen/Core>
#include "abstract_local_optimizer.h"
#include "../utils/types.h"

namespace stereoglue
{
	namespace local_optimization
	{
		// Templated class for estimating a model for RANSAC. This class is purely a
		// virtual class and should be implemented for the specific task that RANSAC is
		// being used for. Two methods must be implemented: estimateModel and residual. All
		// other methods are optional, but will likely enhance the quality of the RANSAC
		// output.
		class LeastSquaresOptimizer : public LocalOptimizer
		{
		protected:
			bool useInliers;

		public:
			LeastSquaresOptimizer() : useInliers(false) {}
			~LeastSquaresOptimizer() {}

			void setUseInliers(const bool kUseInliers_)
			{
				useInliers = kUseInliers_;
			}

			// The function for estimating the model parameters from the data points with multimatches.
			void run(
				const DataMatrix &kDataSrc_,
				const DataMatrix &kDataDst_,
				const DataMatrix &kMatches_,
				const DataMatrix &kMatchScores_,
				const std::vector<std::pair<size_t, size_t>> &kInliers_,
				const models::Model &kModel_,
				const scoring::Score &kScore_,
				const estimator::Estimator *kEstimator_,
				const scoring::AbstractScoring *kScoring_,
				models::Model &estimatedModel_,
				scoring::Score &estimatedScore_,
				std::vector<std::pair<size_t, size_t>> &estimatedInliers_) const
			{
				static const scoring::Score kInvalidScore = scoring::Score();

				// The estimated models
				std::vector<models::Model> estimatedModels;
				scoring::Score currentScore;

				estimatedModels.emplace_back(kModel_);

				if (useInliers)
				{
					// The correspondences for the estimation based on the inliers
					Eigen::MatrixXd correspondences(kInliers_.size(), 4);
					for (size_t i = 0; i < kInliers_.size(); ++i)
						correspondences.row(i) <<
							kDataSrc_(kInliers_[i].first, 0),
							kDataSrc_(kInliers_[i].first, 1),
							kDataDst_(kInliers_[i].second, 0),
							kDataDst_(kInliers_[i].second, 1);

					if (kInliers_.size() > 0)
					{
						// Estimate the model using the inliers
						if (!kEstimator_->estimateModelNonminimal(
							correspondences,  // The data points
							nullptr, 
							kInliers_.size(),
							&estimatedModels,
							nullptr))
						{
							estimatedScore_ = kInvalidScore;
							return;
						}
					}
					else
					{
						// Calculate the score of the estimated model
						currentScore = kScoring_->score(
							kDataSrc_, 
							kDataDst_, 
							kMatches_, 
							kMatchScores_, 
							kModel_, 
							kEstimator_, 
							estimatedInliers_);
							
						// The correspondences for the estimation based on the inliers
						Eigen::MatrixXd correspondences(estimatedInliers_.size(), 4);
						for (size_t i = 0; i < kInliers_.size(); ++i)
							correspondences.row(i) <<
								kDataSrc_(estimatedInliers_[i].first, 0),
								kDataSrc_(estimatedInliers_[i].first, 1),
								kDataDst_(estimatedInliers_[i].second, 0),
								kDataDst_(estimatedInliers_[i].second, 1);

						// Estimate the model using the inliers
						if (!kEstimator_->estimateModelNonminimal(
							correspondences,  // The data points
							nullptr, 
							estimatedInliers_.size(),
							&estimatedModels,
							nullptr))
						{
							estimatedScore_ = kInvalidScore;
							return;
						}
					}
				} else
				{ 
					// The correspondences for the estimation based on the inliers
					//std::vector<double> weights;
					//weights.reserve(kMatches_.rows() * kMatches_.cols());
					Eigen::MatrixXd correspondences(kMatches_.rows() * kMatches_.cols(), 4);
					for (size_t i = 0; i < kMatches_.rows(); ++i)
						for (size_t j = 0; j < kMatches_.cols(); ++j)
							if (kMatches_(i, j) != -1)
							{
								correspondences.row(i * kMatches_.cols() + j) <<
									kDataSrc_(i, 0),
									kDataSrc_(i, 1),
									kDataDst_(kMatches_(i, j), 0),
									kDataDst_(kMatches_(i, j), 1);
								//weights.push_back(kMatchScores_(i, j));
							} else
								break;

					if (!kEstimator_->estimateModelNonminimal(
						correspondences,  // The data points
						nullptr, 
						correspondences.rows(),
						&estimatedModels,
						nullptr))
						//&weights[0]))
					{
						estimatedScore_ = kInvalidScore;
						return;
					}
				}

				// Clear the estimated inliers
				estimatedInliers_.clear();

				// Temp inliers for selecting the best model
				std::vector<std::pair<size_t,size_t>> tmpInliers;
				tmpInliers.reserve(kDataSrc_.rows());

				// Calculate the scoring of the estimated model
				for (const auto &model : estimatedModels)
				{
					// Calculate the score of the estimated model
					tmpInliers.clear();
					currentScore = kScoring_->score(
						kDataSrc_, 
						kDataDst_, 
						kMatches_, 
						kMatchScores_, 
						model, 
						kEstimator_, 
						tmpInliers);

					// Check if the current model is better than the previous one
					if (useInliers || estimatedScore_ < currentScore)
					{
						// Update the estimated model
						estimatedModel_ = model;
						estimatedScore_ = currentScore;
						tmpInliers.swap(estimatedInliers_);
					}
				}
			}

			// The function for estimating the model parameters from the data points.
			void run(const DataMatrix &kData_, // The data points
				const std::vector<size_t> &kInliers_, // The inliers of the previously estimated model
				const models::Model &kModel_, // The previously estimated model 
				const scoring::Score &kScore_, // The of the previously estimated model
				const estimator::Estimator *kEstimator_, // The estimator used for the model estimation
				const scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
				models::Model &estimatedModel_, // The estimated model
				scoring::Score &estimatedScore_, // The score of the estimated model
				std::vector<size_t> &estimatedInliers_) const // The inliers of the estimated model
			{
				static const scoring::Score kInvalidScore = scoring::Score();
				// The estimated models
				std::vector<models::Model> estimatedModels;
				scoring::Score currentScore;

				estimatedModels.emplace_back(kModel_);

				if (useInliers)
				{
					if (kInliers_.size() > 0)
					{
						// Estimate the model using the inliers
						if (!kEstimator_->estimateModelNonminimal(
							kData_,  // The data points
							&kInliers_[0], 
							kInliers_.size(),
							&estimatedModels,
							nullptr))
						{
							estimatedScore_ = kInvalidScore;
							return;
						}
					}
					else
					{
						// Calculate the score of the estimated model
						currentScore = kScoring_->score(kData_, kModel_, kEstimator_, estimatedInliers_);

						// Estimate the model using the inliers
						if (!kEstimator_->estimateModelNonminimal(
							kData_,  // The data points
							&estimatedInliers_[0], 
							estimatedInliers_.size(),
							&estimatedModels,
							nullptr))
						{
							estimatedScore_ = kInvalidScore;
							return;
						}
					}
					//std::cout << " - " << estimatedScore_.getValue() << std::endl;
				} else if (!kEstimator_->estimateModelNonminimal(
					kData_,  // The data points
					nullptr, 
					kData_.rows(),
					&estimatedModels,
					nullptr))
				{
					estimatedScore_ = kInvalidScore;
					return;
				}

				// Clear the estimated inliers
				estimatedInliers_.clear();

				// Temp inliers for selecting the best model
				std::vector<size_t> tmpInliers;
				tmpInliers.reserve(kData_.rows());

				//std::cout << " - " << estimatedModels.size() << std::endl;

				// Calculate the scoring of the estimated model
				for (const auto &model : estimatedModels)
				{
					// Calculate the score of the estimated model
					tmpInliers.clear();
					currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

					// Check if the current model is better than the previous one
					if (useInliers || estimatedScore_ < currentScore)
					{
						// Update the estimated model
						estimatedModel_ = model;
						estimatedScore_ = currentScore;
						tmpInliers.swap(estimatedInliers_);
					}
				}
			}

		};
	}
}  // namespace gcransac