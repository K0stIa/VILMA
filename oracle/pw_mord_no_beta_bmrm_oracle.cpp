/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "pw_mord_no_beta_bmrm_oracle.h"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>
#include <memory>

using namespace BmrmOracle;

template <class Loss>
PwMordNoBetaBmrmOracle<Loss>::PwMordNoBetaBmrmOracle(
    Data *data, const std::vector<int> &cut_labels)
    : PwSingleGenderNoBetaBmrmOracle<Loss>(data, cut_labels) {}

template <class Loss>
PwMordNoBetaBmrmOracle<Loss>::~PwMordNoBetaBmrmOracle() {}

template <class Loss>
double PwMordNoBetaBmrmOracle<Loss>::risk(const double *weights,
                                          double *subgrad) {
  const int nexamples =
      PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataNumExamples();

  const int dim = PwSingleGenderNoBetaBmrmOracle<Loss>::dim;
  double *wx_buffer_ = PwSingleGenderNoBetaBmrmOracle<Loss>::wx_buffer_;
  double *alpha_buffer_ = PwSingleGenderNoBetaBmrmOracle<Loss>::alpha_buffer_;
  const int kPW = PwSingleGenderNoBetaBmrmOracle<Loss>::kPW;
  Data *data_ = PwSingleGenderNoBetaBmrmOracle<Loss>::data_;
  DenseVecD &beta_ = PwSingleGenderNoBetaBmrmOracle<Loss>::beta_;

  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);

  PwSingleGenderNoBetaBmrmOracle<Loss>::ProjectData(params, data_, wx_buffer_,
                                                    kPW);

  gradient.fill(0);

  // TODO: train beta
  static bool first = false;
  if (first) {
    for (int i = 0; i < beta_.dim_; ++i) beta_.data_[i] = 0;
    first = false;
  } else {
    PwMordNoBetaBmrmOracle<Loss>::TrainBeta(&beta_, data_, wx_buffer_,
                                            alpha_buffer_, kPW);
    //    for (int i = 0; i < data_->ny; ++i)
    //      beta_.data_[i] = params.data_[kPW * GetDataDim() + i];
  }

  double *wx = wx_buffer_;
  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val =
        UpdateSingleExampleGradient(params, beta_, wx, example_idx, &gradient);
    obj += val;
    wx += kPW;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

template <class Loss>
double PwMordNoBetaBmrmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &params, const DenseVecD &beta, const double *wx,
    const int example_idx, DenseVecD *gradient) {
  double *alpha_buffer_ = PwSingleGenderNoBetaBmrmOracle<Loss>::alpha_buffer_;
  const int kPW = PwSingleGenderNoBetaBmrmOracle<Loss>::kPW;
  Data *data_ = PwSingleGenderNoBetaBmrmOracle<Loss>::data_;
  Loss &loss_ = PwSingleGenderNoBetaBmrmOracle<Loss>::loss_;

  const int gt_y = data_->y->data_[example_idx];
  auto subproblem =
      PwSingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup222(
          wx, alpha_buffer_, beta.data_, 0,
          PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataNumAgeClasses() - 1,
          gt_y, kPW, &loss_);

  const int best_y = std::get<1>(subproblem);

  // update gradient
  // get reference on curent example
  const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
  // TODO: make more efficient
  {
    double *alpha = alpha_buffer_ + best_y * kPW;
    for (int k = 0; k < kPW; ++k) {
      double *grad = gradient->data_ +
                     k * PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataDim();
      for (int i = 0; i < x->non_zero_; ++i) {
        grad[x->index_[i]] += alpha[k] * x->vals_[i];
      }
    }
  }

  {
    double *alpha = alpha_buffer_ + gt_y * kPW;
    for (int k = 0; k < kPW; ++k) {
      double *grad = gradient->data_ +
                     k * PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataDim();
      for (int i = 0; i < x->non_zero_; ++i) {
        grad[x->index_[i]] -= alpha[k] * x->vals_[i];
      }
    }
  }

  double psi = beta[gt_y];
  {
    double *alpha = alpha_buffer_ + gt_y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  return std::get<0>(subproblem) - psi;
}

//////////////////////////////
template <class Loss>
void PwMordNoBetaBmrmOracle<Loss>::TrainBeta(DenseVecD *beta, Data *data,
                                             double *wx_buffer,
                                             double *alpha_buffer_,
                                             const int kPW) {
  const int num_vars = beta->dim_;

  Accpm::Parameters param;  //(paramFile);

  param.setIntParameter("NumVariables", num_vars);
  param.setIntParameter("NumSubProblems", 1);

  param.setIntParameter("MaxOuterIterations", 2000);
  param.setIntParameter("MaxInnerIterations", 500);

  param.setIntParameter("Proximal", 1);
  param.setRealParameter("Tolerance", 1e-3);
  param.setIntParameter("Verbosity", 0);
  param.setRealParameter("ObjectiveLB", 0.0);

  param.setIntParameter("ConvexityCheck", 1);
  param.setIntParameter("ConvexityFix", 1);

  param.setIntParameter("DynamicRho", 0);

  param.setIntParameter("Ball", 0);
  // param.setRealParameter("RadiusBall", 300);

  param.setRealParameter("WeightEpigraphCutInit", 1);
  param.setRealParameter("WeightEpigraphCutInc", 0);

  param.setVariableLB(vector<double>(num_vars, -100));
  param.setVariableUB(vector<double>(num_vars, 100));

  // param.output(std::cout);

  int n = param.getIntParameter("NumVariables");

  vector<double> start(n, 0);
  param.setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  PwMordAuxiliaryBetaAccpmOracle<Loss> f1(data, wx_buffer, alpha_buffer_, kPW);
  Accpm::Oracle accpm_oracle(&f1);

  Accpm::QpGenerator qpGen;
  qpGen.init(&param, &accpm_oracle);
  while (!qpGen.run()) {
  }
  // qpGen.output(std::cout);

  const Accpm::AccpmVector &x = *qpGen.getQueryPoint();

  for (int i = 0; i < x.size(); ++i) {
    beta->operator[](i) = x(i);
  }

  qpGen.terminate();
}

template <class Loss>
PwMordAuxiliaryBetaAccpmOracle<Loss>::PwMordAuxiliaryBetaAccpmOracle(
    Data *data, double *wx, double *alpha, const int kPW)
    : OracleFunction(),
      loss_(),
      kDim(data->ny),
      kPW(kPW),
      accpm_grad_vector_(data->ny),
      params_(data->ny),
      gradient_(data->ny),
      wx_buffer_(wx),
      alpha_buffer_(alpha),
      data_(data) {}

template <class Loss>
PwMordAuxiliaryBetaAccpmOracle<Loss>::~PwMordAuxiliaryBetaAccpmOracle() {}

template <class Loss>
int PwMordAuxiliaryBetaAccpmOracle<Loss>::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kDim; ++i) {
    params_[i] = y(i);
  }

  const int nexamples = data_->x->kRows;

  gradient_.fill(0);
  double obj = 0;

  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(
        params_, wx_buffer_ + example_idx * kPW, example_idx, &gradient_);
    obj += val;
  }
  // normalize
  gradient_.mul(1. / nexamples);
  obj /= nexamples;

  functionValue = obj;

  for (int i = 0; i < kDim; ++i) {
    accpm_grad_vector_(i) = gradient_[i];
  }

  memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
         sizeof(double) * accpm_grad_vector_.size());

  if (info != nullptr) {
    *info = 1;
  }

  return 0;
}

template <class Loss>
double PwMordAuxiliaryBetaAccpmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, const double *wx, const int example_idx,
    DenseVecD *gradient) {
  // extract example labels
  const int gt_y = data_->y->data_[example_idx];

  const auto subproblem =
      PwSingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup222(
          wx, alpha_buffer_, beta.data_, 0, beta.dim_ - 1, gt_y, kPW, &loss_);

  const int &best_y = std::get<1>(subproblem);

  // update gradient
  gradient->data_[best_y] += 1;
  gradient->data_[gt_y] -= 1;

  double psi = beta[gt_y];
  {
    double *alpha = alpha_buffer_ + gt_y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  return std::get<0>(subproblem) - psi;
}

template class BmrmOracle::PwMordNoBetaBmrmOracle<Vilma::MAELoss>;
template class BmrmOracle::PwMordAuxiliaryBetaAccpmOracle<Vilma::MAELoss>;

template class BmrmOracle::PwMordNoBetaBmrmOracle<Vilma::ZOLoss>;
template class BmrmOracle::PwMordAuxiliaryBetaAccpmOracle<Vilma::ZOLoss>;
