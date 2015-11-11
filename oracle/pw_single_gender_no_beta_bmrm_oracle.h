/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__pw_single_gender_no_beta_bmrm_oracle__
#define __sparse_sgd__pw_single_gender_no_beta_bmrm_oracle__

#include <stdio.h>

#include "../bmrm/bmrm_solver.h"
#include "dense_vector.h"

#include "Oracle.h"

#include <stdio.h>
#include <vector>


struct Data;

namespace BmrmOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class PwSingleGenderNoBetaBmrmOracle : public BMRM_Solver {
 public:
  PwSingleGenderNoBetaBmrmOracle() = delete;

  PwSingleGenderNoBetaBmrmOracle(Data *data,
                                 const std::vector<int> &cut_labels);
  virtual ~PwSingleGenderNoBetaBmrmOracle();

  int GetDataDim();
  int GetOracleParamsDim();
  int GetDataNumExamples();
  int GetDataNumAgeClasses();

  virtual double risk(const double *weights, double *subgrad);

  static std::pair<double, int> SingleExampleBestAgeLabelLookup222(
      const double *wx, const double *alpha, const double *beta, int from,
      int to, const int gt_y, const int kPW, const Loss *const loss_ptr_);

  double EvaluateModel(Data *data, double *params);

  static void ProjectData(const DenseVecD &params, Data *data,
                          double *wx_buffer, const int kPW);
  static void TrainBeta(DenseVecD *beta, Data *data, double *wx_buffer,
                        double *alpha_buffer_, const int kPW);

  const int kPW;
  // store cut labels coef
  double *alpha_buffer_;

 protected:
  using BMRM_Solver::dim;

  /**
   * Single example gradient computation
   */

  double UpdateSingleExampleGradient(const DenseVecD &params,
                                     const DenseVecD &beta, const double *wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

 private:
  Loss loss_;
  // Oracle is never an owner of Data
  Data *data_ = nullptr;
  // buffer to store results <w,x> for all x
  double *wx_buffer_ = nullptr;
  // buffer to store betas
  DenseVecD beta_;
};

template <class Loss>
class PwSingleGenderAuxiliaryBetaAccpmOracle : public Accpm::OracleFunction {
 public:
  PwSingleGenderAuxiliaryBetaAccpmOracle(Data *data, double *wx, double *alpha,
                                         const int kPW);

  ~PwSingleGenderAuxiliaryBetaAccpmOracle();
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

#endif /* defined(__sparse_sgd__pw_single_gender_no_beta_bmrm_oracle__) */
