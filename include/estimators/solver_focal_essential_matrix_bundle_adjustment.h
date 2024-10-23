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

#include "solver_fundamental_matrix_eight_point.h"
#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/camera_pose.h"
#include "numerical_optimizer/essential.h"


namespace stereoglue
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FocalEssentialMatrixBundleAdjustmentSolver : public AbstractSolver
			{
			protected:
				poselib::BundleOptions options;
				size_t pointNumberForCheiralityCheck;

			public:
				FocalEssentialMatrixBundleAdjustmentSolver(poselib::BundleOptions kOptions_ = poselib::BundleOptions())
					: 	options(kOptions_),
						pointNumberForCheiralityCheck(1)
				{
				}

				~FocalEssentialMatrixBundleAdjustmentSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				size_t maximumSolutions() const override
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 6;
				}

				poselib::BundleOptions &getMutableOptions()
				{
					return options;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point

			protected:				
				std::pair<double, double> decomposeFocalLength(const Eigen::Matrix3d& F) const;
			};
			
			std::pair<double, double> FocalEssentialMatrixBundleAdjustmentSolver::decomposeFocalLength(const Eigen::Matrix3d& F) const
			{
				Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
				Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();

				Eigen::Matrix3d F_normalized;
				F_normalized = K2.transpose() * F * K1;

				Eigen::JacobiSVD<Eigen::MatrixXd> svd(F_normalized, Eigen::ComputeThinU | Eigen::ComputeThinV);
				Eigen::MatrixXd U = svd.matrixU();
				Eigen::MatrixXd V = svd.matrixV();

				Eigen::VectorXd e1 = V.col(V.cols() - 1);
				Eigen::VectorXd e2 = U.col(U.cols() - 1);

				double sc1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
				Eigen::MatrixXd R1(3, 3);
				R1 << e1[0] / sc1, e1[1] / sc1, 0, -e1[1] / sc1, e1[0] / sc1, 0, 0, 0, 1;
				Eigen::VectorXd Re1 = R1 * e1;

				double sc2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1]);
				Eigen::MatrixXd R2(3, 3);
				R2 << e2[0] / sc2, e2[1] / sc2, 0, -e2[1] / sc2, e2[0] / sc2, 0, 0, 0, 1;
				Eigen::VectorXd Re2 = R2 * e2;

				Eigen::MatrixXd RF = R2 * F_normalized * R1.transpose();

				Eigen::MatrixXd C2 = Eigen::MatrixXd::Zero(3, 3);
				C2.diagonal() << Re2[2], 1, -Re2[0];
				Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(3, 3);
				C1.diagonal() << Re1[2], 1, -Re1[0];
				Eigen::MatrixXd A = C2.inverse() * RF * C1.inverse();

				double a = A(0, 0);
				double b = A(0, 1);
				double c = A(1, 0);
				double d = A(1, 1);

				double ff1 = -a * c * Re1[0] * Re1[0] / (a * c * Re1[2] * Re1[2] + b * d);
				double f1 = std::sqrt(ff1);

				double ff2 = -a * b * Re2[0] * Re2[0] / (a * b * Re2[2] * Re2[2] + c * d);
				double f2 = std::sqrt(ff2);

				return std::make_pair(f1, f2);
			}

			FORCE_INLINE bool FocalEssentialMatrixBundleAdjustmentSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				// Check if we have enough points for the bundle adjustment
				if (kSampleNumber_ < sampleSize())
					return false;

				// The point correspondences
				std::vector<Eigen::Vector2d> x1(kSampleNumber_); 
				std::vector<Eigen::Vector2d> x2(kSampleNumber_); 
				std::vector<double> weights(kSampleNumber_, 1.0);
				
				// Estimating the essential matrix using the five-point algorithm if no model is provided
				if (models_.size() == 0)
				{
					std::cout << "Error the essential matrix is not provided" << std::endl;
					while (1);
					// Initializing the five-point solver
					EssentialMatrixFivePointNisterSolver fivePointSolver;
					// Estimating the essential matrix
					fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

					// If the estimation failed, return false
					if (models_.size() == 0)
						return false;
				}

				if (kSample_ == nullptr)
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
					}

					if (kWeights_ != nullptr)
						for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
							weights[pointIdx] = kWeights_[pointIdx];
				} else
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(kSample_[pointIdx], 0), kData_(kSample_[pointIdx], 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(kSample_[pointIdx], 2), kData_(kSample_[pointIdx], 3));
					}

					if (kWeights_ != nullptr)
						for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
							weights[pointIdx] = kWeights_[kSample_[pointIdx]];
				}
				
				// The options for the bundle adjustment
				poselib::BundleOptions tmpOptions = options;
				// If the sample is provided, we use a more robust loss function. This typically runs in the end of the robust estimation
				if (kSample_ != nullptr) 
				{
					tmpOptions.loss_scale = 0.5 * options.loss_scale;
					tmpOptions.max_iterations = 100;
					tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
				}
				
				// Select the first point in the sample to be used for the cheirality check
				const size_t kPointNumberForCheck = std::min(pointNumberForCheiralityCheck, kSampleNumber_);
				std::vector<Eigen::Vector3d> x1CheiralityCheck(kPointNumberForCheck), x2CheiralityCheck(kPointNumberForCheck);

				// The pose with the lowest cost
				double bestCost = std::numeric_limits<double>::max();
				poselib::CameraPose bestPose;
				double bestFocalSrc, bestFocalDst;
				std::vector<models::Model> newModels; // The estimated model parameters

				// Iterating through the potential models.
				for (auto& model : models_)
				{
					// Get the fundamental matrix
					Eigen::Matrix3d fundamentalMatrix = model.getMutableData().block<3, 3>(0, 0).eval();

					// Perform the bundle adjustment
					poselib::BundleStats stats;
					poselib::refine_fundamental(
						x1, 
						x2, 
						&fundamentalMatrix,
						tmpOptions);
						
					newModels.resize(newModels.size() + 1);
					newModels.back().getMutableData().resize(3, 3);
					newModels.back().getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;

					// Decompose the fundamental matrix to focal length
					auto focalLengths = decomposeFocalLength(fundamentalMatrix);

					if (std::isnan(focalLengths.first) || std::isnan(focalLengths.second) ||
						std::isinf(focalLengths.first) || std::isinf(focalLengths.second) ||
						focalLengths.first <= 0 || focalLengths.second <= 0)
					{
						// Update the model
						/*newModels.resize(newModels.size() + 1);
						newModels.back().getMutableData().resize(3, 3);
						newModels.back().getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;*/
						continue;
					}

					const double &focalSrc = focalLengths.first;
					const double &focalDst = focalLengths.second;

					Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
					K(0,0) = focalSrc;
					K(1,1) = focalSrc;
					Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
					G(0,0) = focalDst;
					G(1,1) = focalDst;

					Eigen::Matrix3d essentialMatrix = G.inverse() * model.getData().block<3, 3>(0, 0) * K.inverse();
					
					// Decompose the essential matrix to camera poses
					poselib::CameraPoseVector poses;

					poselib::motion_from_essential(
						essentialMatrix, // The essential matrix
						x1CheiralityCheck, x2CheiralityCheck, // The point correspondence used for the cheirality check
						&poses); // The decomposed poses
					
					poselib::Camera 
						cameraSrc("PinholeCameraModel", {0.5 * (focalSrc + focalDst), 0.5 * (focalSrc + focalDst)}, 0, 0), 
						cameraDst("PinholeCameraModel", {0.5 * (focalSrc + focalDst), 0.5 * (focalSrc + focalDst)}, 0, 0);

					poselib::ImagePair imagePair;
					imagePair.camera1 = cameraSrc;
					imagePair.camera2 = cameraDst;
					bool success = false;
					
					// Iterating through the potential poses and optimizing each
					for (auto& pose : poses)
					{
						imagePair.pose = poses[0];

						// Perform the bundle adjustment
						poselib::BundleStats stats;

						/*poselib::refine_relpose(
							x1, 
							x2, 
							&pose,
							tmpOptions);*/

						poselib::refine_shared_focal_relpose(
							x1, 
							x2, 
							&imagePair,
							tmpOptions);

						/*if (stats.cost < bestCost && imagePair.camera1.focal() > 0 && imagePair.camera2.focal() > 0)
						{
							bestCost = stats.cost;
							//bestPose = pose;
							bestPose = imagePair.pose;
							bestFocalSrc = imagePair.camera1.focal();
							bestFocalDst = imagePair.camera2.focal();
						}*/

						if (imagePair.camera1.focal() > 0 && imagePair.camera2.focal() > 0)
						{
							success = true;
							K(0,0) = imagePair.camera1.focal();
							K(1,1) = imagePair.camera1.focal();
							G(0,0) = imagePair.camera2.focal();
							G(1,1) = imagePair.camera2.focal();

							/*std::cout << "Best cost: " << bestCost << std::endl;
							std::cout << "Best focal src: " << bestFocalSrc << std::endl;
							std::cout << "Best focal dst: " << bestFocalDst << std::endl;*/
							
							// Adding the essential matrix as the estimated models.
							newModels.resize(newModels.size() + 1);
							poselib::essential_from_motion(bestPose, &essentialMatrix);
							fundamentalMatrix = G.inverse() * essentialMatrix * K.inverse();
							newModels.back().getMutableData().resize(3, 3);
							newModels.back().getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;
						} //else
						//	model.getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;
					}

					//if (!success)
					{
						newModels.resize(newModels.size() + 1);
						newModels.back().getMutableData().resize(3, 3);
						newModels.back().getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;
					}
				}

				models_ = newModels;
				
				// Composing the essential matrix from the pose
				/*if (bestCost < std::numeric_limits<double>::max())
				{
					Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
					K(0,0) = bestFocalSrc;
					K(1,1) = bestFocalSrc;
					Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
					G(0,0) = bestFocalDst;
					G(1,1) = bestFocalDst;

					Eigen::Matrix3d essentialMatrix;
					poselib::essential_from_motion(bestPose, &essentialMatrix);

					Eigen::Vector3d focals;
					focals << bestFocalSrc, bestFocalDst, 0;

					// Adding the essential matrix as the estimated models.
					models_.resize(1);
					models_[0].getMutableData().resize(3, 4),
					models_[0].getMutableData() << G.inverse() * essentialMatrix * K.inverse(), focals;
					//model.getMutableData()(0, 3) = 1.0;
					//model.getMutableData()(1, 3) = 1.0;

					//std::cout << "Essential matrix: " << std::endl << models_[0].getMutableData() << std::endl << std::endl;
				} else
				{
					for (auto& model : models_)
					{
						// Get the fundamental matrix
						Eigen::Matrix3d fundamentalMatrix = model.getMutableData().block<3, 3>(0, 0).eval();

						// Perform the bundle adjustment
						poselib::BundleStats stats;
						poselib::refine_fundamental(
							x1, 
							x2, 
							&fundamentalMatrix,
							tmpOptions);

						Eigen::Vector3d focals;
						focals << 1.0, 1.0, 0;

						// Update the model
						model.getMutableData().resize(3, 4),
						model.getMutableData() << fundamentalMatrix, focals;
					}
				}*/

				return models_.size() > 0;
			}
		}
	}
}