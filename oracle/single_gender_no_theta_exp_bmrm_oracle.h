/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __vilma__single_gender_no_theta_exp_bmrm_oracle__
#define __vilma__single_gender_no_theta_exp_bmrm_oracle__

#include "../bmrm/bmrm_solver.h"
#include "dense_vector.h"

#include "Oracle.h"

#include <stdio.h>

class Data;

namespace BmrmOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class SingleGenderNoThetaExpBmrmOracle : public BMRM_Solver {
 public:
  SingleGenderNoThetaExpBmrmOracle() = delete;

  SingleGenderNoThetaExpBmrmOracle(Data *data);
  virtual ~SingleGenderNoThetaExpBmrmOracle();

  int GetDataDim();
  int GetOracleParamsDim();
  int GetDataNumExamples();
  int GetDataNumAgeClasses();

  virtual double risk(const double *weights, double *subgrad);

  static int SingleExampleBestAgeLabelLookup(const double wx,
                                             const DenseVecD &theta,
                                             const int ny, const int gt_y);

  static void ProjectData(const DenseVecD &params, Data *data,
                          double *wx_buffer);
  static void TrainTheta(DenseVecD *theta, Data *data, double *wx_buffer);

  double EvaluateModel(Data *data, double *params);

 protected:
  using BMRM_Solver::dim;

  /**
   * Single example gradient computation
   */

  double UpdateSingleExampleGradient(const DenseVecD &params,
                                     const DenseVecD &theta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

 private:
  Loss loss_;
  // Oracle is never an owner of Data
  Data *data_ = nullptr;
  // buffer to store results <w,x> for all x
  double *wx_buffer_ = nullptr;
  // buffer to store thetas
  DenseVecD theta_;
};

template <class Loss>
class SingleGenderAuxiliaryThetaAccpmOracle : public Accpm::OracleFunction {
 public:
  SingleGenderAuxiliaryThetaAccpmOracle(Data *data, double *wx);

  ~SingleGenderAuxiliaryThetaAccpmOracle();
  void UpdateWX(const double *wx);

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info);

 protected:
  double UpdateSingleExampleGradient(const DenseVecD &theta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

  const int kNumAgeClasses;
  DenseVecD params_, gradient_;
  Data *data_;
  Accpm::AccpmVector accpm_grad_vector_;
  double *wx_buffer_;
  Loss loss_;
};
}

#endif /* defined(__vilma__single_gender_no_theta_exp_bmrm_oracle__) */
