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

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace stereoglue
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyOneAffineGravitySolver : public AbstractSolver
			{
			public:
				HomographyOneAffineGravitySolver() : 
					gravitySource(Eigen::Matrix3d::Identity()),
					gravityDestination(Eigen::Matrix3d::Identity())
				{
				}

				~HomographyOneAffineGravitySolver()
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
					return 1;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 2;
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
				Eigen::Matrix3d gravitySource; // The gravity alignment matrix of the source camera
				Eigen::Matrix3d gravityDestination; // The gravity alignment matrix of the destination camera

				FORCE_INLINE bool estimateMinimalModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_) const; // The weight for each point
			};

			FORCE_INLINE bool HomographyOneAffineGravitySolver::estimateMinimalModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_, // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const // The weight for each point
			{
				using namespace Eigen;
				
				const size_t idx = sample_ == nullptr ? 
					0 : sample_[0];

				Eigen::Vector3d p, q;
				p << data_.at<double>(idx, 0), data_.at<double>(idx, 1), 1;
				q << data_.at<double>(idx, 2), data_.at<double>(idx, 3), 1;

				Eigen::Matrix2d A;
				A << data_.at<double>(idx, 4), data_.at<double>(idx, 5),
					data_.at<double>(idx, 6), data_.at<double>(idx, 7);

				const Eigen::Matrix3d &R1 = gravity_source,
					&R2 = gravity_destination;

				//rectify the projections p, q
				Eigen::Vector3d p1 = R1.transpose() * p;
				Eigen::Vector3d q1 = R2.transpose() * q;
				
				//compute the new transformation matrices	
				Eigen::Matrix<double,2,3> B = A.transpose().inverse() * R1.block<2,3>(0,0);
				Eigen::Matrix<double,2,3> C = R2.block<2,3>(0,0);
				
				//build vectors mA, mB, mC, mD, mE, mF, mG, mH, mJ, such that:
				//matrix M(t) = t*[mA mD mG] + (1-t*t)*[mB mE mH] + (1+t*t)*[mC mF mJ]
				//and M(t)*T = 0
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
				
				//the determinant of det(t) of matrix M(t) is a polynomial of degree 6
				//find coefficients c0, c1, c2, c3, c4, c5, c6, such that det(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5 + c6*t^6		
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

						//solve linear equations to get the normal of the plane and complete the homography
						Eigen::Matrix<double,6,3> AA = Eigen::Matrix<double,6,3>::Zero();
						Eigen::Matrix<double,6,1> bb = Eigen::Matrix<double,6,1>::Zero();
						AA(0,0) = p(0)*T(0) - q(0)*p(0)*T(2);
						AA(0,1) = p(1)*T(0) - q(0)*p(1)*T(2);
						AA(0,2) = p(2)*T(0) - q(0)*p(2)*T(2);
						bb(0) = R(0,0)*p(0) + R(0,1)*p(1) + R(0,2)*p(2) - q(0) * ( R(2,0)*p(0) + R(2,1)*p(1) + R(2,2)*p(2) );

						AA(1,0) = p(0)*T(1) - q(1)*p(0)*T(2);
						AA(1,1) = p(1)*T(1) - q(1)*p(1)*T(2);
						AA(1,2) = p(2)*T(1) - q(1)*p(2)*T(2);
						bb(1) = R(1,0)*p(0) + R(1,1)*p(1) + R(1,2)*p(2) - q(1) * ( R(2,0)*p(0) + R(2,1)*p(1) + R(2,2)*p(2) );

						AA(2,0) = T(0) - (q(0)+A(0,0)*p(0))*T(2);
						AA(2,1) = -A(0,0)*p(1)*T(2);
						AA(2,2) = -A(0,0)*T(2);
						bb(2) = R(0,0) - (q(0)+A(0,0)*p(0))*R(2,0) - A(0,0)*p(1)*R(2,1) - A(0,0)*R(2,2);

						AA(3,0) = -A(0,1)*p(0)*T(2);
						AA(3,1) = T(0) - (q(0)+A(0,1)*p(1))*T(2);
						AA(3,2) = -A(0,1)*T(2);
						bb(3) = R(0,1) - (q(0)+A(0,1)*p(1))*R(2,1) - A(0,1)*p(0)*R(2,0) - A(0,1)*R(2,2);

						AA(4,0) = T(1) - (q(1)+A(1,0)*p(0))*T(2);
						AA(4,1) = -A(1,0)*p(1)*T(2);
						AA(4,2) = -A(1,0)*T(2);
						bb(4) = R(1,0) - (q(1)+A(1,0)*p(0))*R(2,0) - A(1,0)*p(1)*R(2,1) - A(1,0)*R(2,2);

						AA(5,0) = -A(1,1)*p(0)*T(2);
						AA(5,1) = T(1) - (q(1)+A(1,1)*p(1))*T(2);
						AA(5,2) = -A(1,1)*T(2);
						bb(5) = R(1,1) - (q(1)+A(1,1)*p(1))*R(2,1) - A(1,1)*p(0)*R(2,0) - A(1,1)*R(2,2);

						Eigen::Vector3d n = AA.colPivHouseholderQr().solve(bb);
						Eigen::Matrix3d H = R - T * n.transpose();

						models::Model model;
						auto &modelData = model.getMutableData();
						modelData = H;
						models_.push_back(model);
					}
				}
				
				return models_.size();
			}

			FORCE_INLINE bool HomographyOneAffineGravitySolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				return estimateMinimalModel(kData_,
					kSample_,
					kSampleNumber_,
					models_,
					kWeights_);
			}
		}
	}
}