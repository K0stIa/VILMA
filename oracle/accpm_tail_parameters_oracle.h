//
//  accpm_tail_parameters_oracle.hpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef accpm_tail_parameters_oracle_h
#define accpm_tail_parameters_oracle_h

#include "dense_vector.h"
#include "ordinal_regression.h"
#include "Oracle.h"

namespace VilmaAccpmOracle {

class AccpmTailParametersOracle : public Accpm::OracleFunction {
 public:
  AccpmTailParametersOracle(VilmaOracle::OrdinalRegression *ord_oracle,
                            double *wx_buffer, const int dim)
      : ord_oracle(ord_oracle),
        kDim(dim),
        accpm_grad_vector_(dim),
        params(dim),
        gradient(dim),
        wx_buffer_(wx_buffer) {}

  virtual ~AccpmTailParametersOracle() = default;

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info);

 protected:
  VilmaOracle::OrdinalRegression *ord_oracle;
  const int kDim;
  Accpm::AccpmVector accpm_grad_vector_;
  Vilma::DenseVector<double> params, gradient;
  double *wx_buffer_;
};
}

#endif /* accpm_tail_parameters_oracle_h */
