/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __vilma__single_gender_no_beta_bmrm_oracle__
#define __vilma__single_gender_no_beta_bmrm_oracle__

#include "../bmrm/bmrm_solver.h"
#include "dense_vector.h"

#include "Oracle.h"

#include <stdio.h>

class Data;

namespace BmrmOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class SingleGenderNoBetaBmrmOracle : public BMRM_Solver {
 public:
  SingleGenderNoBetaBmrmOracle() = delete;

  SingleGenderNoBetaBmrmOracle(Data *data);
  virtual ~SingleGenderNoBetaBmrmOracle();

  int GetDataDim();
  int GetOracleParamsDim();
  int GetDataNumExamples();
  int GetDataNumAgeClasses();

  virtual double risk(const double *weights, double *subgrad);

  static std::tuple<double, int> SingleExampleBestAgeLabelLookup(
      const double wx, const DenseVecD &beta, int from, int to, const int gt_y,
      const Loss *const loss_ptr_);

  static void ProjectData(const DenseVecD &params, Data *data,
                          double *wx_buffer);
  static void TrainBeta(DenseVecD *beta, Data *data, double *wx_buffer);

  double EvaluateModel(Data *data, double *params);

 protected:
  using BMRM_Solver::dim;

  /**
   * Single example gradient computation
   */

  double UpdateSingleExampleGradient(const DenseVecD &params,
                                     const DenseVecD &beta, const double wx,
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
class SingleGenderAuxiliaryBetaAccpmOracle : public Accpm::OracleFunction {
 public:
  SingleGenderAuxiliaryBetaAccpmOracle(Data *data, double *wx);

  ~SingleGenderAuxiliaryBetaAccpmOracle();
  void UpdateWX(const double *wx);

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info);

 protected:
  double UpdateSingleExampleGradient(const DenseVecD &beta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

 private:
  const int kNumAgeClasses;
  DenseVecD params_, gradient_;
  Data *data_;
  Accpm::AccpmVector accpm_grad_vector_;
  double *wx_buffer_;
  Loss loss_;
};
}

#endif /* defined(__vilma__single_gender_no_beta_bmrm_oracle__) */
