//
//  single_gender_no_beta_bmrm_oracle.cpp
//  sparse_sgd
//
//  Created by Kostia on 3/9/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "single_gender_no_beta_bmrm_oracle.h"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>

using namespace BmrmOracle;

template <class Loss>
SingleGenderNoBetaBmrmOracle<Loss>::SingleGenderNoBetaBmrmOracle(Data *data)
    : data_(data), BMRM_Solver(data->x->kCols), beta_(data->ny) {
  wx_buffer_ = new double[data_->x->kRows];
}

template <class Loss>
SingleGenderNoBetaBmrmOracle<Loss>::~SingleGenderNoBetaBmrmOracle() {
  if (wx_buffer_ != nullptr) delete[] wx_buffer_;
}

template <class Loss>
int SingleGenderNoBetaBmrmOracle<Loss>::GetDataDim() {
  return data_->x->kCols;
}

template <class Loss>
int SingleGenderNoBetaBmrmOracle<Loss>::GetOracleParamsDim() {
  return GetDataDim();
}

template <class Loss>
int SingleGenderNoBetaBmrmOracle<Loss>::GetDataNumExamples() {
  return data_->x->kRows;
}

template <class Loss>
int SingleGenderNoBetaBmrmOracle<Loss>::GetDataNumAgeClasses() {
  return data_->ny;
}

template <class Loss>
double SingleGenderNoBetaBmrmOracle<Loss>::risk(const double *weights,
                                                double *subgrad) {
  const int nexamples = GetDataNumExamples();
  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);

  ProjectData(params, data_, wx_buffer_);

  gradient.fill(0);

  // TODO: train beta
  TrainBeta(&beta_, data_, wx_buffer_);

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(
        params, beta_, wx_buffer_[example_idx], example_idx, &gradient);
    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

template <class Loss>
double SingleGenderNoBetaBmrmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &params, const DenseVecD &beta, const double wx,
    const int example_idx, DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
#endif

  const auto left_subproblem =
      SingleExampleBestAgeLabelLookup(wx, beta, 0, gt_yl, gt_yl, &loss_);
  const auto right_subproblem = SingleExampleBestAgeLabelLookup(
      wx, beta, gt_yr, GetDataNumAgeClasses() - 1, gt_yr, &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  // update gradient
  const int coef = best_yl + best_yr - gt_yl - gt_yr;
  // get reference on curent example
  const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
  // TODO: make method for this
  for (int i = 0; i < x->non_zero_; ++i) {
    gradient->data_[x->index_[i]] += coef * x->vals_[i];
  }

  const double psi = wx * (gt_yl + gt_yr) + beta[gt_yl] + beta[gt_yr];
  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template <class Loss>
std::tuple<double, int>
SingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
    const double wx, const DenseVecD &beta, int from, int to, const int gt_y,
    const Loss *const loss_ptr_) {
  double best_cost = 0;
  int best_y = -1;
  for (int y = from; y <= to; ++y) {
    const double cost =
        (loss_ptr_ != nullptr ? loss_ptr_->operator()(y, gt_y) : 0) + wx * y +
        beta[y];
    if (best_y == -1 || best_cost < cost) {
      best_cost = cost;
      best_y = y;
    }
  }
#ifdef USE_ASSERT
  assert(best_y != -1);
#endif
  return std::make_tuple(best_cost, best_y);
}

template <class Loss>
void SingleGenderNoBetaBmrmOracle<Loss>::ProjectData(const DenseVecD &params,
                                                     Data *data,
                                                     double *wx_buffer) {
#ifdef USE_ASSERT
  assert(wx_buffer_ != nullptr);
#endif
  const int nexamples = data->x->kRows;
  // precompute wx[example_idx] = <x, w>
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    const Vilma::SparseVector<double> *x = data->x->GetRow(example_idx);
    wx_buffer[example_idx] = x->dot<DenseVecD>(params);
  }
}

template <class Loss>
void SingleGenderNoBetaBmrmOracle<Loss>::TrainBeta(DenseVecD *beta, Data *data,
                                                   double *wx_buffer) {
  const int num_age_classes = beta->dim_;

  Accpm::Parameters param;  //(paramFile);

  param.setIntParameter("NumVariables", num_age_classes);
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

  param.setVariableLB(vector<double>(num_age_classes, -100));
  param.setVariableUB(vector<double>(num_age_classes, 100));

  // param.output(std::cout);

  int n = param.getIntParameter("NumVariables");

  vector<double> start(n, 0);
  param.setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  SingleGenderAuxiliaryBetaAccpmOracle<Loss> f1(data, wx_buffer);
  Accpm::Oracle accpm_oracle(&f1);

  Accpm::QpGenerator qpGen;
  qpGen.init(&param, &accpm_oracle);
  while (!qpGen.run()) {
  }
  // qpGen.output(std::cout);

  const Accpm::AccpmVector &x = *qpGen.getQueryPoint();

#ifdef USE_ASSERT
  assert(x.size() == GetDataNumAgeClasses());
#endif

  for (int i = 0; i < x.size(); ++i) {
    beta->operator[](i) = x(i);
  }

  qpGen.terminate();
}

template <class Loss>
double SingleGenderNoBetaBmrmOracle<Loss>::EvaluateModel(Data *data,
                                                         double *params) {
  const Data &kData = *data;
  const int num_examples = data->x->kRows;
  const int dim_x = GetDataDim();
  int error = 0;
  DenseVecD beta(data->ny, params + dim_x);
  for (int ex = 0; ex < num_examples; ++ex) {
    const Vilma::SparseVector<double> &x = *kData.x->GetRow(ex);
    const int y = kData.y->data_[ex];
    const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
    std::tuple<double, int> res =
        SingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
            wx, beta, 0, kData.ny - 1, y, nullptr);
    error += loss_(y, std::get<1>(res));
  }

  return 1. * error / num_examples;
}

template <class Loss>
SingleGenderAuxiliaryBetaAccpmOracle<
    Loss>::SingleGenderAuxiliaryBetaAccpmOracle(Data *data, double *wx)
    : OracleFunction(),
      loss_(),
      kNumAgeClasses(data->ny),
      accpm_grad_vector_(data->ny),
      params_(data->ny),
      gradient_(data->ny),
      wx_buffer_(wx),
      data_(data) {}

template <class Loss>
SingleGenderAuxiliaryBetaAccpmOracle<
    Loss>::~SingleGenderAuxiliaryBetaAccpmOracle() {}

template <class Loss>
int SingleGenderAuxiliaryBetaAccpmOracle<Loss>::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kNumAgeClasses; ++i) {
    params_[i] = y(i);
  }
  // TODO: implement this
  // functionValue = oracle_->risk(&params[0], &gradient[0]);

  const int nexamples = data_->x->kRows;

  gradient_.fill(0);
  double obj = 0;

  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(params_, wx_buffer_[example_idx],
                                             example_idx, &gradient_);
    obj += val;
  }
  // normalize
  gradient_.mul(1. / nexamples);
  obj /= nexamples;

  functionValue = obj;

  for (int i = 0; i < kNumAgeClasses; ++i) {
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
double SingleGenderAuxiliaryBetaAccpmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, const double wx, const int example_idx,
    DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
#endif

  const auto left_subproblem =
      SingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
          wx, beta, 0, gt_yl, gt_yl, &loss_);
  const auto right_subproblem =
      SingleGenderNoBetaBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
          wx, beta, gt_yr, data_->ny - 1, gt_yr, &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  // update gradient
  gradient->data_[best_yl] += 1;
  gradient->data_[best_yr] += 1;
  gradient->data_[gt_yl] -= 1;
  gradient->data_[gt_yr] -= 1;

  const double psi = wx * (gt_yl + gt_yr) + beta[gt_yl] + beta[gt_yr];

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template class BmrmOracle::SingleGenderNoBetaBmrmOracle<Vilma::MAELoss>;
template class BmrmOracle::SingleGenderAuxiliaryBetaAccpmOracle<Vilma::MAELoss>;
template class BmrmOracle::SingleGenderNoBetaBmrmOracle<Vilma::ZOLoss>;
template class BmrmOracle::SingleGenderAuxiliaryBetaAccpmOracle<Vilma::ZOLoss>;
