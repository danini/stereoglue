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

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/sturm.h"

namespace stereoglue
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixSingleAffineDepthSolver : public AbstractSolver
			{
			public:
				FundamentalMatrixSingleAffineDepthSolver()
				{
				}

				~FundamentalMatrixSingleAffineDepthSolver()
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
					return 25;
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
				Eigen::MatrixXcd solverUncalibratedAffineDepth(const Eigen::VectorXd& data) const;
				void fastEigenvectorSolver(double * eigv, int neig, Eigen::Matrix<double,9,9> &AM, Eigen::Matrix<std::complex<double>,3,9> &sols) const;
				int solve(const Eigen::Vector3d p, const Eigen::Vector3d q, const Eigen::Matrix2d A1, const Eigen::Matrix2d A2, const double d1, const double d2, const Eigen::Vector2d dd1, const Eigen::Vector2d dd2, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts, double * fs, double * gs) const;
				
				template<typename Derived>
				void charpolyDanilevskyPiv(Eigen::MatrixBase<Derived> &A, double *p) const;

				
			
			};

			Eigen::MatrixXcd FundamentalMatrixSingleAffineDepthSolver::solverUncalibratedAffineDepth(const Eigen::VectorXd& data) const
			{
				// Compute coefficients
				const double* d = data.data();
				Eigen::VectorXd coeffs(24);
				coeffs[0] = d[0];
				coeffs[1] = d[4];
				coeffs[2] = d[1];
				coeffs[3] = d[5];
				coeffs[4] = d[2];
				coeffs[5] = d[6];
				coeffs[6] = d[3];
				coeffs[7] = d[7];
				coeffs[8] = d[8];
				coeffs[9] = d[12];
				coeffs[10] = d[9];
				coeffs[11] = d[13];
				coeffs[12] = d[10];
				coeffs[13] = d[14];
				coeffs[14] = d[11];
				coeffs[15] = d[15];
				coeffs[16] = d[16];
				coeffs[17] = d[20];
				coeffs[18] = d[17];
				coeffs[19] = d[21];
				coeffs[20] = d[18];
				coeffs[21] = d[22];
				coeffs[22] = d[19];
				coeffs[23] = d[23];

				// Setup elimination template
				static const int coeffs0_ind[] = { 0,8,16,0,8,16,1,9,17,0,8,16,2,10,0,8,16,18,1,9,17,0,8,16,2,10,0,8,16,18,3,11,1,9,17,19,1,9,17,2,10,8,0,16,18,4,12,2,10,8,0,16,18,20,3,11,1,9,19,17,1,9,17,8,0,16,2,10,8,0,16,18,5,4,13,3,12,2,11,10,18,20,19,21,3,11,9,1,17,19,9,1,17,4,12,10,2,0,18,8,16,20,6,14,4,12,10,2,18,20,22,5,13,3,11,9,1,17,21,19,3,11,9,19,1,17,10,2,0,18,16,8,7,15,5,4,13,12,10,2,18,20,21,23,6,5,14,4,13,11,12,20,3,19,21,22,11,3,1,19,9,17,6,14,12,4,2,20,10,18,22,6,14,12,4,20,22,7,15,12,4,2,20,18,10,23,14,6,4,22,12,20,7,15,5,13,11,3,19,23,21,7,6,15,13,14,12,5,4,20,21,23,22,5,13,11,21,3,1,19,17,9,6,13,14,22,5,3,21,11,19,14,6,22 };
				static const int coeffs1_ind[] = { 7,23,15,15,7,23,7,15,13,5,21,23,15,14,7,6,4,23,22,20,12,15,7,5,23,21,13,7,15,13,23,5,3,21,19,11,15,14,7,5,23,6,22,13,21,7,6,15,23,22,14,6,14,22 };
				static const int C0_ind[] = { 0,2,35,37,41,66,72,74,107,111,116,142,144,146,148,153,172,179,181,185,210,222,228,247,253,257,259,265,268,282,288,290,291,296,322,323,328,333,352,363,368,370,377,385,394,396,398,400,405,407,414,416,424,431,433,437,438,444,462,463,475,481,484,518,525,533,546,552,555,562,564,571,576,577,578,579,581,583,584,589,592,606,610,611,616,621,622,629,637,640,659,666,668,687,692,694,701,703,709,710,711,718,720,722,724,729,731,738,740,748,755,757,761,762,768,770,777,785,786,787,799,805,807,808,814,816,842,849,851,857,860,861,864,866,867,870,872,876,879,886,888,895,898,899,901,904,905,907,909,910,913,916,917,925,928,930,947,954,955,956,962,963,975,980,982,989,991,997,998,999,1006,1012,1017,1019,1026,1028,1036,1047,1052,1058,1065,1067,1073,1076,1077,1078,1090,1097,1099,1105,1106,1107,1117,1121,1122,1128,1130,1137,1145,1146,1147,1156,1158,1161,1162,1164,1167,1169,1174,1176,1177,1180,1183,1195,1201,1203,1204,1210,1211,1212,1220,1221,1231,1235,1237,1240,1242,1243,1244,1250,1251,1271,1278,1280 } ;
				static const int C1_ind[] = { 23,32,33,50,57,65,78,84,86,93,101,103,118,122,125,129,131,133,137,140,141,159,166,167,168,176,177,187,193,195,196,202,203,204,212,213,227,231,234,235,236,238,240,242,243,271,275,278,279,284,285,307,314,315 };

				Eigen::Matrix<double,36,36> C0; C0.setZero();
				Eigen::Matrix<double,36,9> C1; C1.setZero();
				for (int i = 0; i < 234; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
				for (int i = 0; i < 54; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); } 

				Eigen::Matrix<double,36,9> C12 = C0.partialPivLu().solve(C1);

				// Setup action matrix
				Eigen::Matrix<double,14, 9> RR;
				RR << -C12.bottomRows(5), Eigen::Matrix<double,9,9>::Identity(9, 9);

				static const int AM_ind[] = { 9,7,0,1,10,2,3,11,4 };
				Eigen::Matrix<double, 9, 9> AM;
				for (int i = 0; i < 9; i++) {
					AM.row(i) = RR.row(AM_ind[i]);
				}

				Eigen::Matrix<std::complex<double>, 3, 9> sols;
				sols.setZero();

				// Solve eigenvalue problem
				double p[1+9];
				Eigen::Matrix<double, 9, 9> AMp = AM;
				charpolyDanilevskyPiv(AMp, p);	
				double roots[9];
				int nroots;
				sturm::find_real_roots_sturm(p, 9, roots, &nroots, 8, 0);
				fastEigenvectorSolver(roots, nroots, AM, sols);

				return sols;
			}


			/* Computes characteristic poly using Danilevsky's method with full pivoting */
			template<typename Derived>
			void FundamentalMatrixSingleAffineDepthSolver::charpolyDanilevskyPiv(Eigen::MatrixBase<Derived> &A, double *p) const
			{
				int n = A.rows();

				for (int i = n - 1; i > 0; i--) {

					int piv_ind = i - 1;
					double piv = std::abs(A(i, i - 1));

					// Find largest pivot
					for (int j = 0; j < i - 1; j++) {
						if (std::abs(A(i, j)) > piv) {
							piv = std::abs(A(i, j));
							piv_ind = j;
						}
					}
					if (piv_ind != i - 1) {
						// Perform permutation
						A.row(i - 1).swap(A.row(piv_ind));
						A.col(i - 1).swap(A.col(piv_ind));
					}
					piv = A(i, i - 1);

					Eigen::VectorXd v = A.row(i);
					A.row(i - 1) = v.transpose()*A;

					Eigen::VectorXd vinv = (-1.0)*v;
					vinv(i - 1) = 1;
					vinv /= piv;
					vinv(i - 1) -= 1;
					Eigen::VectorXd Acol = A.col(i - 1);
					for (int j = 0; j <= i; j++)
						A.row(j) = A.row(j) + Acol(j)*vinv.transpose();


					A.row(i) = Eigen::VectorXd::Zero(n);
					A(i, i - 1) = 1;
				}
				p[n] = 1;
				for (int i = 0; i < n; i++)
					p[i] = -A(0, n - i - 1);
			}

			void FundamentalMatrixSingleAffineDepthSolver::fastEigenvectorSolver(
				double * eigv, 
				int neig, 
				Eigen::Matrix<double,9,9> &AM, 
				Eigen::Matrix<std::complex<double>,3,9> &sols) const
			{
				static const int ind[] = { 2,3,5,6,8 };	
				// Truncated action matrix containing non-trivial rows
				Eigen::Matrix<double, 5, 9> AMs;
				double zi[3];
				
				for (int i = 0; i < 5; i++)	
					AMs.row(i) = AM.row(ind[i]);
				
				for (int i = 0; i < neig; i++) {
					zi[0] = eigv[i];
					for (int j = 1; j < 3; j++)
					{
						zi[j] = zi[j - 1] * eigv[i];
					}
					Eigen::Matrix<double, 5,5> AA;
					AA.col(0) = AMs.col(3);
					AA.col(1) = AMs.col(1) + zi[0] * AMs.col(2);
					AA.col(2) = AMs.col(8);
					AA.col(3) = zi[0] * AMs.col(6) + AMs.col(7);
					AA.col(4) = AMs.col(0) + zi[0] * AMs.col(4) + zi[1] * AMs.col(5);
					AA(1,0) = AA(1,0) - zi[0];
					AA(0,1) = AA(0,1) - zi[1];
					AA(4,2) = AA(4,2) - zi[0];
					AA(3,3) = AA(3,3) - zi[1];
					AA(2,4) = AA(2,4) - zi[2];

					Eigen::Matrix<double, 4, 1>  s = AA.leftCols(4).colPivHouseholderQr().solve(-AA.col(4));
					sols(0,i) = s(1);
					sols(1,i) = zi[0];
					sols(2,i) = s(3);
				}
			}

			int FundamentalMatrixSingleAffineDepthSolver::solve(const Eigen::Vector3d p, const Eigen::Vector3d q, const Eigen::Matrix2d A1, const Eigen::Matrix2d A2, const double d1, const double d2, const Eigen::Vector2d dd1, const Eigen::Vector2d dd2, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts, double * fs, double * gs) const
			{
				double x1 = p(0);
				double y1 = p(1);
				double x2 = q(0);
				double y2 = q(1);

				const double c1 = dd1.transpose()*A1.col(0);
				const double c2 = d1*A1(0,0);
				const double c3 = d1*A1(1,0);
				
				const double c4 = dd1.transpose()*A1.col(1);
				const double c5 = d1*A1(0,1);
				const double c6 = d1*A1(1,1);
				
				const double c7 = dd2.transpose()*A2.col(0);
				const double c8 = d2*A2(0,0);
				const double c9 = d2*A2(1,0);
				
				const double c10 = dd2.transpose()*A2.col(1);
				const double c11 = d2*A2(0,1);
				const double c12 = d2*A2(1,1);
				
				const double w11_0 = x1*x1*x1*x1*x1*x1 + x1*x1*x1*x1*y1*y1 + y1*y1*x1*x1*x1*x1 + y1*y1*x1*x1*y1*y1 + x1*x1*y1*y1*x1*x1 + x1*x1*y1*y1*y1*y1 + y1*y1*y1*y1*x1*x1 + y1*y1*y1*y1*y1*y1;
				const double w11_1 = x1*x1*x1*x1 + y1*y1*x1*x1 + x1*x1*x1*x1 + x1*x1*y1*y1 + x1*x1*y1*y1 + y1*y1*y1*y1 + y1*y1*x1*x1 + y1*y1*y1*y1 + x1*x1*x1*x1 + x1*x1*y1*y1 + y1*y1*x1*x1 + y1*y1*y1*y1;
				const double w11_2 = x1*x1 + y1*y1 + x1*x1 + y1*y1 + x1*x1 + y1*y1;
				const double w11_3 = 1;
				
				const double w22_0 = y1*y1*y1*y1 + x1*x1*y1*y1;
				const double w22_1 = y1*y1 + y1*y1 + x1*x1;
				const double w22_2 = 1;
				
				const double w33_0 = x1*x1*y1*y1 + x1*x1*x1*x1;
				const double w33_1 = x1*x1 + x1*x1 + y1*y1;
				const double w33_2 = 1;
				
				const double w23_0 = -y1*y1*x1*y1 - x1*y1*x1*x1;
				const double w23_1 = -x1*y1;
				
				const double v11_0 = c1*c1*w11_0 + c2*c2*w22_0 + 2*c2*c3*w23_0 + c3*c3*w33_0;
				const double v11_1 = c1*c1*w11_1 + c2*c2*w22_1 + 2*c2*c3*w23_1 + c3*c3*w33_1;
				const double v11_2 = c1*c1*w11_2 + c2*c2*w22_2 + c3*c3*w33_2;
				const double v11_3 = c1*c1*w11_3;
				
				const double v22_0 = c4*c4*w11_0 + c5*c5*w22_0 + 2*c5*c6*w23_0 + c6*c6*w33_0;
				const double v22_1 = c4*c4*w11_1 + c5*c5*w22_1 + 2*c5*c6*w23_1 + c6*c6*w33_1;
				const double v22_2 = c4*c4*w11_2 + c5*c5*w22_2 + c6*c6*w33_2;
				const double v22_3 = c4*c4*w11_3;
				
				const double v12_0 = c1*c4*w11_0 + c2*c5*w22_0 + c2*c6*w23_0 + c5*c3*w23_0 + c3*c6*w33_0;
				const double v12_1 = c1*c4*w11_1 + c2*c5*w22_1 + c2*c6*w23_1 + c5*c3*w23_1 + c3*c6*w33_1;
				const double v12_2 = c1*c4*w11_2 + c2*c5*w22_2 + c3*c6*w33_2;
				const double v12_3 = c1*c4*w11_3;
				
				const double w44_0 = x2*x2*x2*x2*x2*x2 + x2*x2*x2*x2*y2*y2 + y2*y2*x2*x2*x2*x2 + y2*y2*x2*x2*y2*y2 + x2*x2*y2*y2*x2*x2 + x2*x2*y2*y2*y2*y2 + y2*y2*y2*y2*x2*x2 + y2*y2*y2*y2*y2*y2;
				const double w44_1 = x2*x2*x2*x2 + y2*y2*x2*x2 + x2*x2*x2*x2 + x2*x2*y2*y2 + x2*x2*y2*y2 + y2*y2*y2*y2 + y2*y2*x2*x2 + y2*y2*y2*y2 + x2*x2*x2*x2 + x2*x2*y2*y2 + y2*y2*x2*x2 + y2*y2*y2*y2;
				const double w44_2 = x2*x2 + y2*y2 + x2*x2 + y2*y2 + x2*x2 + y2*y2;
				const double w44_3 = 1;

				const double w55_0 = y2*y2*y2*y2 + x2*x2*y2*y2;
				const double w55_1 = y2*y2 + y2*y2 + x2*x2;
				const double w55_2 = 1;

				const double w66_0 = x2*x2*y2*y2 + x2*x2*x2*x2;
				const double w66_1 = x2*x2 + x2*x2 + y2*y2;
				const double w66_2 = 1;

				const double w56_0 = -y2*y2*x2*y2 - x2*y2*x2*x2;
				const double w56_1 = -x2*y2;

				const double v33_0 = c7*c7*w44_0 + c8*c8*w55_0 + 2*c8*c9*w56_0 + c9*c9*w66_0;
				const double v33_1 = c7*c7*w44_1 + c8*c8*w55_1 + 2*c8*c9*w56_1 + c9*c9*w66_1;
				const double v33_2 = c7*c7*w44_2 + c8*c8*w55_2 + c9*c9*w66_2;
				const double v33_3 = c7*c7*w44_3;

				const double v44_0 = c10*c10*w44_0 + c11*c11*w55_0 + 2*c11*c12*w56_0 + c12*c12*w66_0;
				const double v44_1 = c10*c10*w44_1 + c11*c11*w55_1 + 2*c11*c12*w56_1 + c12*c12*w66_1;
				const double v44_2 = c10*c10*w44_2 + c11*c11*w55_2 + c12*c12*w66_2;
				const double v44_3 = c10*c10*w44_3;

				const double v34_0 = c7*c10*w44_0 + c8*c11*w55_0 + c8*c12*w56_0 + c11*c9*w56_0 + c9*c12*w66_0;
				const double v34_1 = c7*c10*w44_1 + c8*c11*w55_1 + c8*c12*w56_1 + c11*c9*w56_1 + c9*c12*w66_1;
				const double v34_2 = c7*c10*w44_2 + c8*c11*w55_2 + c9*c12*w66_2;
				const double v34_3 = c7*c10*w44_3;

				Eigen::VectorXd data(24);
				data(0) = v11_3/v11_0;
				data(1) = v11_2/v11_0;
				data(2) = v11_1/v11_0;
				data(3) = v11_0/v11_0;
				data(4) = v33_3/v33_0;
				data(5) = v33_2/v33_0;
				data(6) = v33_1/v33_0;
				data(7) = v33_0/v33_0;
				
				data(8) = v22_3/v11_0;
				data(9) = v22_2/v11_0;
				data(10) = v22_1/v11_0;
				data(11) = v22_0/v11_0;
				data(12) = v44_3/v33_0;
				data(13) = v44_2/v33_0;
				data(14) = v44_1/v33_0;
				data(15) = v44_0/v33_0;
				
				data(16) = v12_3/v11_0;
				data(17) = v12_2/v11_0;
				data(18) = v12_1/v11_0;
				data(19) = v12_0/v11_0;
				data(20) = v34_3/v33_0;
				data(21) = v34_2/v33_0;
				data(22) = v34_1/v33_0;
				data(23) = v34_0/v33_0;
				
				Eigen::MatrixXcd sol = solverUncalibratedAffineDepth(data);

				const Eigen::Matrix2d A = A2*A1.inverse();
				const double a1 = A(0,0);
				const double a2 = A(0,1);
				const double a3 = A(1,0);
				const double a4 = A(1,1);

				const double p0 = p(0);
				const double p1 = p(1);
				const double q0 = q(0);
				const double q1 = q(1);

				int num_sols = 0;
				bool pseudo_left = 0;
				bool pseudo_right = 0;
				bool pseudo_both = 0;
				
				double f_s[25] = {0.75,0.75,0.75,0.75,0.75,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,2,2,2,2,2,2.8,2.8,2.8,2.8,2.8};
				double g_s[25] = {0.75,1,1.5,2,2.8,0.75,1,1.5,2,2.8,0.75,1,1.5,2,2.8,0.75,1,1.5,2,2.8,0.75,1,1.5,2,2.8};
				for(int i = 0; i < 25; ++i)
				{
					//compute the focal length
					const double f = f_s[i];
					const double g = g_s[i];
					fs[num_sols] = f;
					gs[num_sols] = g;

					//construct the intrinsics matrices
					Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
					K(0,0) = f;
					K(1,1) = f;
					const Eigen::Matrix3d Ki = K.inverse();

					Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
					G(0,0) = g;
					G(1,1) = g;
					const Eigen::Matrix3d Gi = G.inverse();

					//construct the directions in 3D
					Eigen::Matrix<double, 3,2> dx1;
					double div1 = std::sqrt(p(0)*p(0)+p(1)*p(1)+f*f);
					div1 = div1*div1*div1;
					dx1(0,0) = (p(1)*p(1)+f*f)/div1;
					dx1(0,1) = -(p(0)*p(1))/div1;
					dx1(1,0) = -(p(0)*p(1))/div1;
					dx1(1,1) = (p(0)*p(0)+f*f)/div1;
					dx1(2,0) = -(f*p(0))/div1;
					dx1(2,1) = -(f*p(1))/div1;
					
					Eigen::Matrix<double, 3,2> dx2;
					double div2 = std::sqrt(q(0)*q(0)+q(1)*q(1)+g*g);
					div2 = div2*div2*div2;
					dx2(0,0) = (q(1)*q(1)+g*g)/div2;
					dx2(0,1) = -(q(0)*q(1))/div2;
					dx2(1,0) = -(q(0)*q(1))/div2;
					dx2(1,1) = (q(0)*q(0)+g*g)/div2;
					dx2(2,0) = -(g*q(0))/div2;
					dx2(2,1) = -(g*q(1))/div2;
				
					//left camera matrix formulation
					Eigen::Vector3d kp1 = K.inverse()*p;
					kp1 = kp1/kp1.norm();
					Eigen::Matrix<double,3,2> DX1 = (kp1*dd1.transpose() + d1*dx1)*A1;
					
					//right camera matrix formulation
					Eigen::Vector3d kq1 = G.inverse()*q;
					kq1 = kq1/kq1.norm();
					Eigen::Matrix<double,3,2> DX2 = (kq1*dd2.transpose() + d2*dx2)*A2;

					Eigen::Vector3d d1 = DX1.col(0).normalized();
					Eigen::Vector3d d2 = DX1.col(1).normalized();
					Eigen::Vector3d d3 = (d1.cross(d2)).normalized();

					Eigen::Vector3d e1 = DX2.col(0).normalized();
					Eigen::Vector3d e2 = DX2.col(1).normalized();
					Eigen::Vector3d e3 = (e1.cross(e2)).normalized();

					Eigen::Matrix3d E;
					E.col(0) = e1;
					E.col(1) = e2;
					E.col(2) = e3;

					Eigen::Matrix3d D;
					D.col(0) = d1;
					D.col(1) = d2;
					D.col(2) = d3;

					//find the rotation matrix
					Eigen::Matrix3d R0 = E*D.inverse();
					if(R0.determinant() < 0)
						R0 = -1*R0;

					Eigen::JacobiSVD<Eigen::Matrix3d> USV_R(R0, Eigen::ComputeFullU | Eigen::ComputeFullV);
					Eigen::Matrix3d R = USV_R.matrixU() * USV_R.matrixV().transpose();

					Rs[num_sols] = R;

					//find the translation
					const double c1t0 = f*g*R(1,2)+g*R(1,0)*p0+g*R(1,1)*p1-f*R(2,2)*q1-R(2,0)*p0*q1-R(2,1)*p1*q1;
					const double c1t1 = -f*g*R(0,2)-g*R(0,0)*p0-g*R(0,1)*p1+f*R(2,2)*q0+R(2,0)*p0*q0+R(2,1)*p1*q0;
					const double c1t2 = -f*R(1,2)*q0-R(1,0)*p0*q0-R(1,1)*p1*q0+f*R(0,2)*q1+R(0,0)*p0*q1+R(0,1)*p1*q1;

					const double c2t0 = -f*R(2,2)*a3-R(2,0)*p0*a3-R(2,1)*p1*a3+g*R(1,0)-R(2,0)*q1;
					const double c2t1 = f*R(2,2)*a1+R(2,0)*p0*a1+R(2,1)*p1*a1-g*R(0,0)+R(2,0)*q0;
					const double c2t2 = -f*R(1,2)*a1-R(1,0)*p0*a1-R(1,1)*p1*a1+f*R(0,2)*a3+R(0,0)*p0*a3+R(0,1)*p1*a3-R(1,0)*q0+R(0,0)*q1;

					const double c3t0 = -f*R(2,2)*a4-R(2,0)*p0*a4-R(2,1)*p1*a4+g*R(1,1)-R(2,1)*q1;
					const double c3t1 = f*R(2,2)*a2+R(2,0)*p0*a2+R(2,1)*p1*a2-g*R(0,1)+R(2,1)*q0;
					const double c3t2 = -f*R(1,2)*a2-R(1,0)*p0*a2-R(1,1)*p1*a2+f*R(0,2)*a4+R(0,0)*p0*a4+R(0,1)*p1*a4-R(1,1)*q0+R(0,1)*q1;

					Eigen::Matrix3d M;
					M << c1t0,c1t1,c1t2, c2t0,c2t1,c2t2, c3t0,c3t1,c3t2;

					Eigen::JacobiSVD<Eigen::Matrix3d> USV(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
					Eigen::Matrix3d V = USV.matrixV();
					Eigen::Vector3d t = V.col(2);
					t = t/t.norm();

					Ts[num_sols] = t;

					++num_sols;
				}

				return num_sols;
			}

			FORCE_INLINE bool FundamentalMatrixSingleAffineDepthSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{
				double weight = 1.0;
				const size_t idx = kSample_ == nullptr ? 
					0 : kSample_[0];

				const Eigen::Vector3d p(kData_(idx, 0), kData_(idx, 1), 1);
				const Eigen::Vector3d q(kData_(idx, 2), kData_(idx, 3), 1);

				const double
					d1 = kData_(idx, 8),
					d2 = kData_(idx, 9);

				const Eigen::Matrix2d A1 = Eigen::Matrix2d::Identity();
				Eigen::Matrix2d A2;
				A2 << kData_(idx, 4), kData_(idx, 5), 
					kData_(idx, 6), kData_(idx, 7);

				const Eigen::Vector2d dd1(kData_(idx, 10), kData_(idx, 11));
				const Eigen::Vector2d dd2(kData_(idx, 12), kData_(idx, 13));

				Eigen::Matrix3d Rs[25];
				Eigen::Vector3d Ts[25];
				double fs[25];
				double gs[25];
				const int num_sols = solve(p, q, A1, A2, d1, d2, dd1, dd2, Rs, Ts, fs, gs);
				if (!num_sols)
					return false;

				for(size_t i = 0; i < num_sols; ++i)
				{
					Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
					K(0,0) = fs[i];
					K(1,1) = fs[i];
					Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
					G(0,0) = gs[i];
					G(1,1) = gs[i];
					Eigen::Matrix3d Tx = Eigen::Matrix3d::Zero();
					Tx << 0,-Ts[i](2),Ts[i](1), Ts[i](2),0,-Ts[i](0), -Ts[i](1),Ts[i](0),0;

					Eigen::Matrix3d F = G.inverse()*Tx*Rs[i]*K.inverse();
					// Eigen::Matrix3d E = Tx * Rs[i];

					Eigen::Vector3d focals;
					focals << fs[i], gs[i], 0;

					models::Model model;
					auto &modelData = model.getMutableData();
					modelData.resize(3, 4);
					//modelData << E, focals;
					//std::cout << modelData << std::endl;
					//std::cout << "--------------------------" << std::endl;
					modelData << F, focals;
					models_.emplace_back(model);
				}

				return true;
			}
		}
	}
}
