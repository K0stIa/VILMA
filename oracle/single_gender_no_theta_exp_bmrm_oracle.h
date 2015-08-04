//
//  single_gender_no_theta_exp_bmrm_oracle.h
//  vilma
//
//  Created by Kostia on 8/4/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

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

  static std::tuple<double, int> SingleExampleBestAgeLabelLookup(
      const double wx, const DenseVecD &theta, int from, int to, const int gt_y,
      const Loss *const loss_ptr_);

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

 private:
  const int kNumAgeClasses;
  DenseVecD params_, gradient_;
  Data *data_;
  Accpm::AccpmVector accpm_grad_vector_;
  double *wx_buffer_;
  Loss loss_;
};
}

#endif /* defined(__vilma__single_gender_no_theta_exp_bmrm_oracle__) */
