/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__pw_mord_no_beta_bmrm_oracle__
#define __sparse_sgd__pw_mord_no_beta_bmrm_oracle__

#include <stdio.h>
#include <vector>

#include "../bmrm/bmrm_solver.h"
#include "pw_single_gender_no_beta_bmrm_oracle.h"
#include "dense_vector.h"

#include "Oracle.h"

struct Data;

namespace BmrmOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class PwMordNoBetaBmrmOracle : public PwSingleGenderNoBetaBmrmOracle<Loss> {
 public:
  PwMordNoBetaBmrmOracle() = delete;

  PwMordNoBetaBmrmOracle(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwMordNoBetaBmrmOracle();

  virtual double risk(const double *weights, double *subgrad);

  static void TrainBeta(DenseVecD *beta, Data *data, double *wx_buffer,
                        double *alpha_buffer_, const int kPW);

 protected:
  /**
   * Single example gradient computation
   */

  double UpdateSingleExampleGradient(const DenseVecD &params,
                                     const DenseVecD &beta, const double *wx,
                                     const int example_idx,
                                     DenseVecD *gradient);
};

template <class Loss>
class PwMordAuxiliaryBetaAccpmOracle : public Accpm::OracleFunction {
 public:
  PwMordAuxiliaryBetaAccpmOracle(Data *data, double *wx, double *alpha,
                                 const int kPW);

  ~PwMordAuxiliaryBetaAccpmOracle();
  void UpdateWX(const double *wx);

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info);

 protected:
  double UpdateSingleExampleGradient(const DenseVecD &beta, const double *wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

 private:
  const int kDim, kPW;
  DenseVecD params_, gradient_;
  Data *data_;
  Accpm::AccpmVector accpm_grad_vector_;
  double *wx_buffer_;
  double *alpha_buffer_;
  Loss loss_;
};
}

#endif /* defined(__sparse_sgd__pw_mord_no_beta_bmrm_oracle__) */
