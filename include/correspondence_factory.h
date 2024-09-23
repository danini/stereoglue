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

#include <Eigen/Core>

#include "estimators/abstract_estimator.h"
#include "samplers/abstract_sampler.h"
#include "scoring/abstract_scoring.h"
#include "local_optimization/abstract_local_optimizer.h"
#include "termination/abstract_criterion.h"
#include "settings.h"
#include "utils/types.h"

namespace stereoglue {

class AbstractCorrespondenceFactory
{
    public:
        // The dimension of the correspondence
        virtual size_t dimensions() const = 0;

        // Create the correspondence from a given index
        virtual void create(
            const Eigen::MatrixXd &kDataSrc_,
            const Eigen::MatrixXd &kDataDst_,
            const size_t kIdxSrc_,
            const size_t kIdxDst_,
            Eigen::MatrixXd &kCorrespondence_) const = 0;
};

class AffineCorrespondenceFactory : public AbstractCorrespondenceFactory
{
    // The dimension of the correspondence
    size_t dimensions() const override
    {
        return 8;
    }

    // Create the correspondence from a given index
    void create(
        const Eigen::MatrixXd &kDataSrc_,
        const Eigen::MatrixXd &kDataDst_,
        const size_t kIdxSrc_,
        const size_t kIdxDst_,
        Eigen::MatrixXd &kCorrespondence_) const override
    {
        const double &x1 = kDataSrc_(kIdxSrc_, 0),
            &y1 = kDataSrc_(kIdxSrc_, 1),
            &x2 = kDataDst_(kIdxDst_, 0),
            &y2 = kDataDst_(kIdxDst_, 1);

        Eigen::Matrix2d A1, A2, A;

        A1 << kDataSrc_(kIdxSrc_, 2), kDataSrc_(kIdxSrc_, 3),
            kDataSrc_(kIdxSrc_, 4), kDataSrc_(kIdxSrc_, 5);

        A2 << kDataDst_(kIdxDst_, 2), kDataDst_(kIdxDst_, 3),
            kDataDst_(kIdxDst_, 4), kDataDst_(kIdxDst_, 5);

        A = A2 * A1.inverse();

        kCorrespondence_(0) = x1;
        kCorrespondence_(1) = y1;
        kCorrespondence_(2) = x2;
        kCorrespondence_(3) = y2;
        kCorrespondence_(4) = A(0, 0);
        kCorrespondence_(5) = A(0, 1);
        kCorrespondence_(6) = A(1, 0);
        kCorrespondence_(7) = A(1, 1);
    }
};

}
