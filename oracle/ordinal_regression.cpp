/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "data.h"

#include "ordinal_regression.h"

using namespace VilmaOracle;

OrdinalRegression::OrdinalRegression(Data *data)
    : BMRM_Solver(data->GetDataDim() + data->GetDataNumClasses() - 1),
      data_(data) {
  wx_buffer_.reset(new double[data->GetDataNumExamples()]);
}

int OrdinalRegression::GetOracleParamsDim() { return dim; }

void OrdinalRegression::ProjectData(const DenseVecD &w, Data *data,
                                    double *wx_buffer) {
  const int nexamples = data->GetDataNumExamples();
  // precompute wx[example_idx] = <x, w>
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    const Vilma::SparseVector<double> *x = data->x->GetRow(example_idx);
    wx_buffer[example_idx] = x->dot<DenseVecD>(w);
  }
}

int OrdinalRegression::SingleExampleBestLabelLookup(
    const double wx, const DenseVecD &theta, const int data_num_classes) {
  int best_y = 0;
  for (int y = 0; y < data_num_classes - 1; ++y) {
    if (wx >= theta[y]) {
      ++best_y;
    }
  }
  return best_y;
}

std::vector<double> OrdinalRegression::Train() { return BMRM_Solver::learn(); }

double OrdinalRegression::risk(const double *weights, double *subgrad) {
  std::fill(subgrad, subgrad + GetOracleParamsDim(), 0);

  const int nexamples = data_->GetDataNumExamples();

  DenseVecD w(data_->GetDataDim(), const_cast<double *>(weights));
  DenseVecD theta(data_->GetDataNumClasses() - 1,
                  const_cast<double *>(weights) + data_->GetDataDim());

  ProjectData(w, data_, wx_buffer_.get());

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = this->UpdateSingleExampleGradient(
        theta, wx_buffer_[example_idx], example_idx, subgrad,
        subgrad + data_->GetDataDim());
    obj += val;
  }
  // normalize
  for (int i = 0; i < GetOracleParamsDim(); ++i) subgrad[i] /= nexamples;
  obj /= nexamples;

  return obj;
}

Data *OrdinalRegression::GetOracleData() { return data_; }

double *OrdinalRegression::GetWxBuffer() { return wx_buffer_.get(); }