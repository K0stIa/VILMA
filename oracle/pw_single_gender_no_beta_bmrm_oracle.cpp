/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "pw_single_gender_no_beta_bmrm_oracle.h"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>
#include <memory>
#include <algorithm>

using namespace BmrmOracle;

template <class Loss>
PwSingleGenderNoBetaBmrmOracle<Loss>::PwSingleGenderNoBetaBmrmOracle(
    Data *data, const std::vector<int> &cut_labels)
    : data_(data),
      kPW((int)cut_labels.size()),
      BMRM_Solver(data->x->kCols * (int)cut_labels.size()),
      beta_(data->ny) {
  wx_buffer_ = new double[data_->x->kRows * kPW];
  alpha_buffer_ = new double[kPW * (data->ny + 1)];

  std::fill(alpha_buffer_, alpha_buffer_ + kPW * data->ny, 0);
  std::fill(wx_buffer_, wx_buffer_ + kPW * data->ny, 0);

  // init alpha

  const int ny = data->ny;
  assert(kPW >= 1);

  for (int z = 0; z < kPW - 1; ++z) {
    int y1 = cut_labels[z];
    int y2 = cut_labels[z + 1];
    int n = y2 - y1 + 1;
    for (int i = 0; i < n; ++i) {
      double alpha = 1.0 * (n - i - 1) / (n - 1);
      alpha_buffer_[z + (y1 + i) * kPW] = alpha;
      alpha_buffer_[z + 1 + (y1 + i) * kPW] = 1.0 - alpha;
    }
  }

  if (kPW == 1) {
    for (int i = 0; i < ny; ++i) alpha_buffer_[i] = 1.0;
  }
}

template <class Loss>
PwSingleGenderNoBetaBmrmOracle<Loss>::~PwSingleGenderNoBetaBmrmOracle() {
  if (wx_buffer_ != nullptr) delete[] wx_buffer_;
  if (alpha_buffer_ != nullptr) delete[] alpha_buffer_;
}

template <class Loss>
int PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataDim() {
  return data_->x->kCols;
}

template <class Loss>
int PwSingleGenderNoBetaBmrmOracle<Loss>::GetOracleParamsDim() {
  return GetDataDim() * kPW;
}

template <class Loss>
int PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataNumExamples() {
  return data_->x->kRows;
}

template <class Loss>
int PwSingleGenderNoBetaBmrmOracle<Loss>::GetDataNumAgeClasses() {
  return data_->ny;
}

template <class Loss>
double PwSingleGenderNoBetaBmrmOracle<Loss>::risk(const double *weights,
                                                  double *subgrad) {
  const int nexamples = GetDataNumExamples();

  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);

  ProjectData(params, data_, wx_buffer_, kPW);

  gradient.fill(0);

  TrainBeta(&beta_, data_, wx_buffer_, alpha_buffer_, kPW);

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
void PwSingleGenderNoBetaBmrmOracle<Loss>::ProjectData(const DenseVecD &params,
                                                       Data *data,
                                                       double *wx_buffer,
                                                       const int kPW) {
  const int dim_x = data->x->kCols;
  // create weight components
  vector<std::unique_ptr<DenseVecD>> w(kPW);
  for (int i = 0; i < kPW; ++i) {
    w[i].reset(new DenseVecD(dim_x, params.data_ + dim_x * i));
  }

  const int nexamples = data->x->kRows;
  // precompute wx[example_idx] = <x, w>
  int pos = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    const Vilma::SparseVector<double> *x = data->x->GetRow(example_idx);
    for (int k = 0; k < kPW; ++k) {
      wx_buffer[pos++] = x->dot<DenseVecD>(*w[k]);
    }
  }
}

template <class Loss>
double PwSingleGenderNoBetaBmrmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &params, const DenseVecD &beta, const double *wx,
    const int example_idx, DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < data->ny);
  assert(0 <= yr && yr < data->ny);
  assert(yl <= yr);
#endif

  const auto left_subproblem = SingleExampleBestAgeLabelLookup222(
      wx, alpha_buffer_, beta.data_, 0, gt_yl, gt_yl, kPW, &loss_);
  const auto right_subproblem = SingleExampleBestAgeLabelLookup222(
      wx, alpha_buffer_, beta.data_, gt_yr, GetDataNumAgeClasses() - 1, gt_yr,
      kPW, &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  // update gradient
  // get reference on curent example
  const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
  // TODO: make more efficient
  for (int y : {best_yl, best_yr}) {
    double *alpha = alpha_buffer_ + y * kPW;
    for (int k = 0; k < kPW; ++k) {
      double *grad = gradient->data_ + k * GetDataDim();
      for (int i = 0; i < x->non_zero_; ++i) {
        grad[x->index_[i]] += alpha[k] * x->vals_[i];
      }
    }
  }

  for (int y : {gt_yl, gt_yr}) {
    double *alpha = alpha_buffer_ + y * kPW;
    for (int k = 0; k < kPW; ++k) {
      double *grad = gradient->data_ + k * GetDataDim();
      for (int i = 0; i < x->non_zero_; ++i) {
        grad[x->index_[i]] -= alpha[k] * x->vals_[i];
      }
    }
  }

  double psi = beta[gt_yl] + beta[gt_yr];
  for (int y : {gt_yl, gt_yr}) {
    double *alpha = alpha_buffer_ + y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  //  double *beta_grad = gradient->data_ + GetDataDim() * kPW;
  //  beta_grad[best_yl] += 1;
  //  beta_grad[best_yr] += 1;
  //  beta_grad[gt_yl] -= 1;
  //  beta_grad[gt_yr] -= 1;

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template <class Loss>
std::pair<double, int>
PwSingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup222(
    const double *wx, const double *alpha, const double *beta, int from, int to,
    const int gt_y, const int kPW, const Loss *const loss_ptr_) {
  double best_cost = 0;
  int best_y = -1;
  for (int y = from; y <= to; ++y) {
    double cost = beta[y];
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      cost += wx[k] * alpha[y * kPW + k];
    }
    if (loss_ptr_ != nullptr) {
      cost += loss_ptr_->operator()(y, gt_y);
    }
    if (best_y == -1 || best_cost < cost) {
      best_cost = cost;
      best_y = y;
    }
  }
#ifdef USE_ASSERT
  assert(best_y != -1);
#endif
  return std::make_pair(best_cost, best_y);
}

/////////////////////////////
template <class Loss>
void PwSingleGenderNoBetaBmrmOracle<Loss>::TrainBeta(DenseVecD *beta,
                                                     Data *data,
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

  PwSingleGenderAuxiliaryBetaAccpmOracle<Loss> f1(data, wx_buffer,
                                                  alpha_buffer_, kPW);
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
double PwSingleGenderNoBetaBmrmOracle<Loss>::EvaluateModel(Data *data,
                                                           double *params) {
  const int num_examples = data->x->kRows;
  const int dim_x = data->x->kCols;
  const int ny = data_->ny;
  int error = 0;

  vector<double> wx(dim_x * kPW);
  DenseVecD weights(dim_x * kPW, params);
  PwSingleGenderNoBetaBmrmOracle<Loss>::ProjectData(weights, data, &wx[0], kPW);
  const double *beta = params + dim_x * kPW;

  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    const int gt_y = data->y->operator[](example_idx);

    const auto subproblem_res = SingleExampleBestAgeLabelLookup222(
        &wx[0] + example_idx * kPW, alpha_buffer_, beta, 0, ny - 1, -1, kPW,
        nullptr);

    const int &best_y = std::get<1>(subproblem_res);

    error += loss_(gt_y, best_y);
  }

  return 1. * error / num_examples;
}

template <class Loss>
PwSingleGenderAuxiliaryBetaAccpmOracle<
    Loss>::PwSingleGenderAuxiliaryBetaAccpmOracle(Data *data, double *wx,
                                                  double *alpha, const int kPW)
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
PwSingleGenderAuxiliaryBetaAccpmOracle<
    Loss>::~PwSingleGenderAuxiliaryBetaAccpmOracle() {}

template <class Loss>
int PwSingleGenderAuxiliaryBetaAccpmOracle<Loss>::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kDim; ++i) {
    params_[i] = y(i);
  }
  // TODO: implement this
  // functionValue = oracle_->risk(&params[0], &gradient[0]);

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
double
PwSingleGenderAuxiliaryBetaAccpmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, const double *wx, const int example_idx,
    DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < data_->ny);
  assert(0 <= yr && yr < data_->ny);
  assert(yl <= yr);
#endif

  const auto left_subproblem =
      PwSingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup222(
          wx, alpha_buffer_, beta.data_, 0, gt_yl, gt_yl, kPW, &loss_);
  const auto right_subproblem =
      PwSingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup222(
          wx, alpha_buffer_, beta.data_, gt_yr, data_->ny - 1, gt_yr, kPW,
          &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  // update gradient
  gradient->data_[best_yl] += 1;
  gradient->data_[best_yr] += 1;
  gradient->data_[gt_yl] -= 1;
  gradient->data_[gt_yr] -= 1;

  double psi = beta[gt_yl] + beta[gt_yr];
  for (int y : {gt_yl, gt_yr}) {
    double *alpha = alpha_buffer_ + y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template class BmrmOracle::PwSingleGenderNoBetaBmrmOracle<Vilma::MAELoss>;
template class BmrmOracle::PwSingleGenderAuxiliaryBetaAccpmOracle<
    Vilma::MAELoss>;

template class BmrmOracle::PwSingleGenderNoBetaBmrmOracle<Vilma::ZOLoss>;
template class BmrmOracle::PwSingleGenderAuxiliaryBetaAccpmOracle<
    Vilma::ZOLoss>;
