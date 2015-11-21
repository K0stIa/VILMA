/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef accpm_tail_parameters_oracle_h
#define accpm_tail_parameters_oracle_h

#include "dense_vector.h"
#include "ordinal_regression.h"
#include "Oracle.h"

namespace VilmaAccpmOracle {

class AccpmTailParametersOracle : public Accpm::OracleFunction {
 public:
  AccpmTailParametersOracle(VilmaOracle::OrdinalRegression *ord_oracle,
                            double *wx_buffer, const int dim, const int dw = 1)
      : ord_oracle(ord_oracle),
        kDim(dim),
        accpm_grad_vector_(dim),
        params(dim),
        gradient(dim),
        wx_buffer_(wx_buffer),
        kDw(dw) {}

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
  const int kDw;
};
}

#endif /* accpm_tail_parameters_oracle_h */
