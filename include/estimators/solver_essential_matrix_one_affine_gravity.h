// Copyright (C) 2019 Czech Technical University.
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

#include <Eigen/Eigen>
#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/sturm.h"
#include "numerical_optimizer/essential.h"
#include "unsupported/Eigen/Polynomials"

namespace stereoglue
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating an essential matrix between two images
			// when we are given the gravity direction in the two images.
			class EssentialMatrixOneAffineGravity : public AbstractSolver
			{
			public:
				EssentialMatrixOneAffineGravity() : 
					gravitySource(Eigen::Matrix3d::Identity()),
					gravityDestination(Eigen::Matrix3d::Identity())
				{
				}

				~EssentialMatrixOneAffineGravity()
				{
				}

				void setGravity(const Eigen::Matrix3d &kGravitySource,
								const Eigen::Matrix3d &kGravityDestination_)
				{
					gravitySource = kGravitySource;
					gravityDestination = kGravityDestination_;
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
					return 10;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 1;
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
				Eigen::Matrix3d gravitySource;
				Eigen::Matrix3d gravityDestination;

				Eigen::MatrixXcd solver3ptCaliess(const Eigen::VectorXd &data_) const;
			};

			FORCE_INLINE bool EssentialMatrixOneAffineGravity::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{
				using namespace Eigen;
				
				const size_t idx = kSample_ == nullptr ? 
					0 : kSample_[0];

				Eigen::Vector3d p, q;
				p << kData_(idx, 0), kData_(idx, 1), 1;
				q << kData_(idx, 2), kData_(idx, 3), 1;

				Eigen::Matrix2d A;
				A << kData_(idx, 4), kData_(idx, 5),
					kData_(idx, 6), kData_(idx, 7);

				const Eigen::Matrix3d &R1 = gravitySource,
					&R2 = gravityDestination;

				// rectify the projections p, q
				Eigen::Vector3d p1 = R1.transpose() * p;
				Eigen::Vector3d q1 = R2.transpose() * q;
				
				// compute the new transformation matrices	
				Eigen::Matrix<double, 2, 3> B = A.transpose().inverse() * R1.block<2,3>(0,0);
				Eigen::Matrix<double, 2, 3> C = R2.block<2,3>(0,0);
					
				// build vectors mA, mB, mC, mD, mE, mF, mG, mH, mJ, such that:
				// matrix M(t) = t*[mA mD mG] + (1-t*t)*[mB mE mH] + (1+t*t)*[mC mF mJ]
				// and M(t)*T = 0
				Eigen::Vector3d mA;
				mA << p1(0)*q1(1)*2, B(0,0)*q1(1)*2 + C(0,1)*p1(0)*2, B(1,0)*q1(1)*2 + C(1,1)*p1(0)*2;
				Eigen::Vector3d mB;
				mB << p1(2)*q1(1), B(0,2)*q1(1) + C(0,1)*p1(2), B(1,2)*q1(1) + C(1,1)*p1(2);
				Eigen::Vector3d mC;
				mC << -p1(1)*q1(2), -B(0,1)*q1(2) - C(0,2)*p1(1), -B(1,1)*q1(2) - C(1,2)*p1(1);
				
				Eigen::Vector3d mD;
				mD << -p1(0)*q1(0)*2 - p1(2)*q1(2)*2, -B(0,0)*q1(0)*2 - B(0,2)*q1(2)*2 - C(0,0)*p1(0)*2 - C(0,2)*p1(2)*2, -B(1,0)*q1(0)*2 - B(1,2)*q1(2)*2 - C(1,0)*p1(0)*2 - C(1,2)*p1(2)*2;
				Eigen::Vector3d mE;
				mE << - p1(2)*q1(0) + p1(0)*q1(2), B(0,0)*q1(2) - B(0,2)*q1(0) - C(0,0)*p1(2) + C(0,2)*p1(0), B(1,0)*q1(2) - B(1,2)*q1(0) - C(1,0)*p1(2) + C(1,2)*p1(0);
				//mF = [0 0 0]^T
				
				Eigen::Vector3d mG;
				mG << p1(2)*q1(1)*2, B(0,2)*q1(1)*2 + C(0,1)*p1(2)*2, B(1,2)*q1(1)*2 + C(1,1)*p1(2)*2;
				Eigen::Vector3d mH;
				mH << -p1(0)*q1(1), -B(0,0)*q1(1) - C(0,1)*p1(0), -B(1,0)*q1(1) - C(1,1)*p1(0);
				Eigen::Vector3d mJ;
				mJ << p1(1)*q1(0), B(0,1)*q1(0) + C(0,0)*p1(1), B(1,1)*q1(0) + C(1,0)*p1(1);
				
				// the determinant of det(t) of matrix M(t) is a polynomial of degree 6
				// find coefficients c0, c1, c2, c3, c4, c5, c6, such that det(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5 + c6*t^6		
				const double m01 = mA(0)*mD(1)*mG(2);
				const double m02 = mA(0)*mD(1)*mH(2) + mA(0)*mE(1)*mG(2) + mB(0)*mD(1)*mG(2);
				const double m03 = mA(0)*mD(1)*mJ(2) + mC(0)*mD(1)*mG(2);
				const double m04 = mA(0)*mE(1)*mH(2) + mB(0)*mD(1)*mH(2) + mB(0)*mE(1)*mG(2);
				const double m05 = mA(0)*mE(1)*mJ(2) + mB(0)*mD(1)*mJ(2) + mC(0)*mD(1)*mH(2) + mC(0)*mE(1)*mG(2);
				const double m06 = mC(0)*mD(1)*mJ(2);
				const double m07 = mB(0)*mE(1)*mH(2);
				const double m08 = mB(0)*mE(1)*mJ(2) + mC(0)*mE(1)*mH(2);
				const double m09 = mC(0)*mE(1)*mJ(2);
				
				const double m11 = -mA(0)*mD(2)*mG(1);
				const double m12 = -mA(0)*mD(2)*mH(1) - mA(0)*mE(2)*mG(1) - mB(0)*mD(2)*mG(1);
				const double m13 = -mA(0)*mD(2)*mJ(1) - mC(0)*mD(2)*mG(1);
				const double m14 = -mA(0)*mE(2)*mH(1) - mB(0)*mD(2)*mH(1) - mB(0)*mE(2)*mG(1);
				const double m15 = -mA(0)*mE(2)*mJ(1) - mB(0)*mD(2)*mJ(1) - mC(0)*mD(2)*mH(1) - mC(0)*mE(2)*mG(1);
				const double m16 = -mC(0)*mD(2)*mJ(1);
				const double m17 = -mB(0)*mE(2)*mH(1);
				const double m18 = -mB(0)*mE(2)*mJ(1) - mC(0)*mE(2)*mH(1);
				const double m19 = -mC(0)*mE(2)*mJ(1);
				
				const double m21 = -mA(1)*mD(0)*mG(2);
				const double m22 = -mA(1)*mD(0)*mH(2) - mA(1)*mE(0)*mG(2) - mB(1)*mD(0)*mG(2);
				const double m23 = -mA(1)*mD(0)*mJ(2) - mC(1)*mD(0)*mG(2);
				const double m24 = -mA(1)*mE(0)*mH(2) - mB(1)*mD(0)*mH(2) - mB(1)*mE(0)*mG(2);
				const double m25 = -mA(1)*mE(0)*mJ(2) - mB(1)*mD(0)*mJ(2) - mC(1)*mD(0)*mH(2) - mC(1)*mE(0)*mG(2);
				const double m26 = -mC(1)*mD(0)*mJ(2);
				const double m27 = -mB(1)*mE(0)*mH(2);
				const double m28 = -mB(1)*mE(0)*mJ(2) - mC(1)*mE(0)*mH(2);
				const double m29 = -mC(1)*mE(0)*mJ(2);
				
				const double m31 = mA(1)*mD(2)*mG(0);
				const double m32 = mA(1)*mD(2)*mH(0) + mA(1)*mE(2)*mG(0) + mB(1)*mD(2)*mG(0);
				const double m33 = mA(1)*mD(2)*mJ(0) + mC(1)*mD(2)*mG(0);
				const double m34 = mA(1)*mE(2)*mH(0) + mB(1)*mD(2)*mH(0) + mB(1)*mE(2)*mG(0);
				const double m35 = mA(1)*mE(2)*mJ(0) + mB(1)*mD(2)*mJ(0) + mC(1)*mD(2)*mH(0) + mC(1)*mE(2)*mG(0);
				const double m36 = mC(1)*mD(2)*mJ(0);
				const double m37 = mB(1)*mE(2)*mH(0);
				const double m38 = mB(1)*mE(2)*mJ(0) + mC(1)*mE(2)*mH(0);
				const double m39 = mC(1)*mE(2)*mJ(0);
				
				const double m41 = mA(2)*mD(0)*mG(1);
				const double m42 = mA(2)*mD(0)*mH(1) + mA(2)*mE(0)*mG(1) + mB(2)*mD(0)*mG(1);
				const double m43 = mA(2)*mD(0)*mJ(1) + mC(2)*mD(0)*mG(1);
				const double m44 = mA(2)*mE(0)*mH(1) + mB(2)*mD(0)*mH(1) + mB(2)*mE(0)*mG(1);
				const double m45 = mA(2)*mE(0)*mJ(1) + mB(2)*mD(0)*mJ(1) + mC(2)*mD(0)*mH(1) + mC(2)*mE(0)*mG(1);
				const double m46 = mC(2)*mD(0)*mJ(1);
				const double m47 = mB(2)*mE(0)*mH(1);
				const double m48 = mB(2)*mE(0)*mJ(1) + mC(2)*mE(0)*mH(1);
				const double m49 = mC(2)*mE(0)*mJ(1);
				
				const double m51 = -mA(2)*mD(1)*mG(0);
				const double m52 = -mA(2)*mD(1)*mH(0) - mA(2)*mE(1)*mG(0) - mB(2)*mD(1)*mG(0);
				const double m53 = -mA(2)*mD(1)*mJ(0) - mC(2)*mD(1)*mG(0);
				const double m54 = -mA(2)*mE(1)*mH(0) - mB(2)*mD(1)*mH(0) - mB(2)*mE(1)*mG(0);
				const double m55 = -mA(2)*mE(1)*mJ(0) - mB(2)*mD(1)*mJ(0) - mC(2)*mD(1)*mH(0) - mC(2)*mE(1)*mG(0);
				const double m56 = -mC(2)*mD(1)*mJ(0);
				const double m57 = -mB(2)*mE(1)*mH(0);
				const double m58 = -mB(2)*mE(1)*mJ(0) - mC(2)*mE(1)*mH(0);
				const double m59 = -mC(2)*mE(1)*mJ(0);
				
				const double c0 = m07 + m08 + m09 + m17 + m18 + m19 + m27 + m28 + m29 + m37 + m38 + m39 + m47 + m48 + m49 + m57 + m58 + m59;
				const double c1 = m04 + m05 + m06 + m14 + m15 + m16 + m24 + m25 + m26 + m34 + m35 + m36 + m44 + m45 + m46 + m54 + m55 + m56;
				const double c2 = m02+m03-3*m07-m08+m09 + m12+m13-3*m17-m18+m19 + m22+m23-3*m27-m28+m29 + m32+m33-3*m37-m38+m39 + m42+m43-3*m47-m48+m49 + m52+m53-3*m57-m58+m59;
				const double c3 = m01-2*m04+2*m06 + m11-2*m14+2*m16 + m21-2*m24+2*m26 + m31-2*m34+2*m36 + m41-2*m44+2*m46 + m51-2*m54+2*m56;
				const double c4 = -m02+m03+3*m07-m08-m09 -m12+m13+3*m17-m18-m19 -m22+m23+3*m27-m28-m29 -m32+m33+3*m37-m38-m39 -m42+m43+3*m47-m48-m49 -m52+m53+3*m57-m58-m59;
				const double c5 = m04 - m05 + m06 + m14 - m15 + m16 + m24 - m25 + m26 + m34 - m35 + m36 + m44 - m45 + m46 + m54 - m55 + m56;
				const double c6 = -m07 + m08 - m09 - m17 + m18 - m19 - m27 + m28 - m29 - m37 + m38 - m39 - m47 + m48 - m49 - m57 + m58 - m59;
				
				//solve equation det(t)=0 by a companion matrix approach
				Eigen::Matrix<double,6,6> CM = Eigen::Matrix<double,6,6>::Zero();
				CM(1,0) = 1;
				CM(2,1) = 1;
				CM(3,2) = 1;
				CM(4,3) = 1;
				CM(5,4) = 1;
				
				CM(0,5) = -c0/c6;
				CM(1,5) = -c1/c6;
				CM(2,5) = -c2/c6;
				CM(3,5) = -c3/c6;
				CM(4,5) = -c4/c6;
				CM(5,5) = -c5/c6;
				
				const Eigen::MatrixXcd evs = CM.eigenvalues();
				
				//find the real solutions, build the relative poses and unrectify them
				int num_real_sols = 0;
				for(int i=0; i<6; ++i)
				{
					if(evs(i).imag() < 1e-10 && evs(i).imag() > -1e-10)
					{
						//real solution
						const double t = evs(i).real();
						
						//build the rotation matrix of the rectified problem
						Eigen::Matrix3d RR;
						RR << (1-t*t)/(1+t*t), 0, -2*t/(1+t*t), 0, 1, 0, 2*t/(1+t*t), 0, (1-t*t)/(1+t*t);
						
						//find the translation of the rectified problem
						Eigen::Matrix3d M;
						M(0,0) = p1(0)*q1(1)*2*t + p1(2)*q1(1)*(1-t*t) - p1(1)*q1(2)*(1+t*t);
						M(0,1) = -p1(0)*q1(0)*2*t - p1(2)*q1(0)*(1-t*t) + p1(0)*q1(2)*(1-t*t) - p1(2)*q1(2)*2*t;
						M(0,2) = p1(1)*q1(0)*(1+t*t) - p1(0)*q1(1)*(1-t*t) + p1(2)*q1(1)*2*t;
						
						M(1,0) = B(0,0)*q1(1)*2*t - B(0,1)*q1(2)*(1+t*t) + B(0,2)*q1(1)*(1-t*t) + C(0,1)*p1(0)*2*t + C(0,1)*p1(2)*(1-t*t) - C(0,2)*p1(1)*(1+t*t);
						M(1,1) = -B(0,0)*q1(0)*2*t + B(0,0)*q1(2)*(1-t*t) - B(0,2)*q1(0)*(1-t*t) - B(0,2)*q1(2)*2*t - C(0,0)*p1(0)*2*t - C(0,0)*p1(2)*(1-t*t) + C(0,2)*p1(0)*(1-t*t) - C(0,2)*p1(2)*2*t;
						M(1,2) = -B(0,0)*q1(1)*(1-t*t) + B(0,1)*q1(0)*(1+t*t) + B(0,2)*q1(1)*2*t + C(0,0)*p1(1)*(1+t*t) - C(0,1)*p1(0)*(1-t*t) + C(0,1)*p1(2)*2*t;
						
						M(2,0) = B(1,0)*q1(1)*2*t - B(1,1)*q1(2)*(1+t*t) + B(1,2)*q1(1)*(1-t*t) + C(1,1)*p1(0)*2*t + C(1,1)*p1(2)*(1-t*t) - C(1,2)*p1(1)*(1+t*t);
						M(2,1) = -B(1,0)*q1(0)*2*t + B(1,0)*q1(2)*(1-t*t) - B(1,2)*q1(0)*(1-t*t) - B(1,2)*q1(2)*2*t - C(1,0)*p1(0)*2*t - C(1,0)*p1(2)*(1-t*t) + C(1,2)*p1(0)*(1-t*t) - C(1,2)*p1(2)*2*t;
						M(2,2) = -B(1,0)*q1(1)*(1-t*t) + B(1,1)*q1(0)*(1+t*t) + B(1,2)*q1(1)*2*t + C(1,0)*p1(1)*(1+t*t) - C(1,1)*p1(0)*(1-t*t) + C(1,1)*p1(2)*2*t;
						
						
						Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
						const Eigen::Vector3d TR = svd.matrixV().col(2);
						
						//find the rectified pose
						Eigen::Matrix3d R = R2*RR*R1.transpose();
						Eigen::Vector3d T = R2*TR;

						// The cross product matrix of the translation vector
						Eigen::Matrix3d cross_prod_t_dst_src;
						cross_prod_t_dst_src << 
							0, -T(2), T(1), 
							T(2), 0, -T(0),
							-T(1), T(0), 0;
								
						models::Model model;
						auto &modelData = model.getMutableData();
						modelData = cross_prod_t_dst_src * R;
						models_.emplace_back(model);
					}
				}
				
				return models_.size();
			}
		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
