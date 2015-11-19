/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "svor_imc.hpp"

#include "Parameters.h"
#include "QpGenerator.h"
#include "data.h"
#include "loss.h"

using namespace BmrmOracle;

template <class Loss>
SvorImc<Loss>::SvorImc(Data *data)
    : SvorImcReg<Loss>(data), theta_(data->ny - 1) {
  // this setup works fine due to BMRM immplementation
  dim = GetOracleParamsDim();
}

template <class Loss>
int SvorImc<Loss>::GetOracleParamsDim() {
  return GetDataDim();
}

template <class Loss>
double SvorImc<Loss>::risk(const double *weights, double *subgrad) {
  const int nexamples = GetDataNumExamples();
  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);

  ProjectData(params, data_, wx_buffer_.get());

  gradient.fill(0);

  // TODO: train theta
  TrainTheta(&theta_, data_, wx_buffer_.get());

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(theta_, wx_buffer_[example_idx],
                                             example_idx, &gradient);
    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

template <class Loss>
double SvorImc<Loss>::UpdateSingleExampleGradient(const DenseVecD &theta,
                                                  const double wx,
                                                  const int example_idx,
                                                  DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  double obj = 0;
  int total_sgn = 0;

  for (int y = 0; y < data_->ny - 1; ++y) {
    if (gt_yl <= y && y < gt_yr) {
      continue;
    }
    int sgn = 0;
    if (y < gt_yl) {
      sgn = 1;
    }
    if (gt_yr <= y) {
      sgn = -1;
    }
    double score = 1.0 - (wx - theta[y]) * sgn;
    if (score > 0) {
      obj += score;
      total_sgn += sgn;
    }
  }

  if (total_sgn != 0) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // update data cmponent of gradient
    for (int i = 0; i < x->non_zero_; ++i) {
      gradient->data_[x->index_[i]] -= total_sgn * x->vals_[i];
    }
  }

  return obj;
}

template <class Loss>
void SvorImc<Loss>::TrainTheta(DenseVecD *theta, Data *data,
                               double *wx_buffer) {
  const int var_dim = theta->dim_;

  Accpm::Parameters param;  //(paramFile);

  param.setIntParameter("NumVariables", var_dim);
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

  param.setVariableLB(vector<double>(var_dim, -100));
  param.setVariableUB(vector<double>(var_dim, 100));

  // param.output(std::cout);

  int n = param.getIntParameter("NumVariables");

  vector<double> start(n, 0);
  param.setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  SvorImcAuxiliaryThetaAccpmOracle<Loss> f1(data, wx_buffer);
  Accpm::Oracle accpm_oracle(&f1);

  Accpm::QpGenerator qpGen;
  qpGen.init(&param, &accpm_oracle);
  while (!qpGen.run()) {
  }
  // qpGen.output(std::cout);

  const Accpm::AccpmVector &x = *qpGen.getQueryPoint();

  for (int i = 0; i < x.size(); ++i) {
    theta->operator[](i) = x(i);
  }

  qpGen.terminate();
}

template <class Loss>
SvorImcAuxiliaryThetaAccpmOracle<Loss>::SvorImcAuxiliaryThetaAccpmOracle(
    Data *data, double *wx)
    : OracleFunction(),
      loss_(),
      kNumAgeClasses(data->ny),
      accpm_grad_vector_(data->ny - 1),
      params_(data->ny - 1),
      gradient_(data->ny - 1),
      wx_buffer_(wx),
      data_(data) {}

template <class Loss>
SvorImcAuxiliaryThetaAccpmOracle<Loss>::~SvorImcAuxiliaryThetaAccpmOracle() {}

template <class Loss>
int SvorImcAuxiliaryThetaAccpmOracle<Loss>::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kNumAgeClasses - 1; ++i) {
    params_[i] = y(i);
  }

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

  for (int i = 0; i < kNumAgeClasses - 1; ++i) {
    accpm_grad_vector_(i) = gradient_[i];
  }

  if (info != nullptr) {
    *info = 1;
  }

  memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
         sizeof(double) * accpm_grad_vector_.size());

  return 0;
}

template <class Loss>
double SvorImcAuxiliaryThetaAccpmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &theta, const double wx, const int example_idx,
    DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  double obj = 0;
  int total_sgn = 0;

  for (int y = 0; y < data_->ny - 1; ++y) {
    if (gt_yl <= y && y < gt_yr) {
      continue;
    }
    int sgn = 0;
    if (y < gt_yl) {
      sgn = 1;
    }
    if (gt_yr <= y) {
      sgn = -1;
    }
    double score = 1.0 - (wx - theta[y]) * sgn;
    if (score > 0) {
      obj += score;
      gradient->data_[y] += sgn;
      total_sgn += sgn;
    }
  }

  return obj;
}

template class BmrmOracle::SvorImc<Vilma::MAELoss>;
template class BmrmOracle::SvorImcAuxiliaryThetaAccpmOracle<Vilma::MAELoss>;
template class BmrmOracle::SvorImc<Vilma::ZOLoss>;
template class BmrmOracle::SvorImcAuxiliaryThetaAccpmOracle<Vilma::ZOLoss>;