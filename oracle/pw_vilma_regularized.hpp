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

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>
#include <memory>

#include "pw_vilma_regularized.h"

template <class Loss>
VilmaOracle::PwVilmaRegularized<Loss>::PwVilmaRegularized(
    Data *data, const std::vector<int> &cut_labels)
    : OrdinalRegression(data), kPW((int)cut_labels.size()) {
  // setup BMRM dim
  dim = GetOracleParamsDim();

  wx_buffer_.reset(new double[data_->GetDataNumExamples() * kPW]);
  alpha_buffer_.reset(BuildAlphas(cut_labels, data_->GetDataNumClasses()));
}

template <class Loss>
double *VilmaOracle::PwVilmaRegularized<Loss>::BuildAlphas(
    const std::vector<int> &cut_labels, const int ny) {
  const int kPW = (int)cut_labels.size();
  double *alpha_buffer = new double[kPW * (ny + 1)];
  std::fill(alpha_buffer, alpha_buffer + kPW * (ny + 1), 0);

  for (int z = 0; z < kPW - 1; ++z) {
    int y1 = cut_labels[z];
    int y2 = cut_labels[z + 1];
    int n = y2 - y1 + 1;
    for (int i = 0; i < n; ++i) {
      double alpha = 1.0 * (n - i - 1) / (n - 1);
      alpha_buffer[z + (y1 + i) * kPW] = alpha;
      alpha_buffer[z + 1 + (y1 + i) * kPW] = 1.0 - alpha;
    }
  }

  if (kPW == 1) {
    for (int i = 0; i < ny; ++i) alpha_buffer[i] = 1.0;
  }

  return alpha_buffer;
}

template <class Loss>
int VilmaOracle::PwVilmaRegularized<Loss>::GetOracleParamsDim() {
  return data_->GetDataDim() * kPW + this->GetFreeParamsDim();
}

template <class Loss>
int VilmaOracle::PwVilmaRegularized<Loss>::GetFreeParamsDim() {
  return data_->GetDataNumClasses();
}

template <class Loss>
double VilmaOracle::PwVilmaRegularized<Loss>::risk(const double *weights,
                                                   double *subgrad) {
  const int pw_dim = data_->GetDataDim() * kPW;
  std::fill(subgrad, subgrad + GetOracleParamsDim(), 0);

  const int nexamples = data_->GetDataNumExamples();

  DenseVecD w(pw_dim, const_cast<double *>(weights));
  // free_params is theta for Ord and beta for Mord
  DenseVecD free_params(this->GetFreeParamsDim(),
                        const_cast<double *>(weights) + pw_dim);

  ProjectData(w, data_, wx_buffer_.get(), kPW);

  double *wx = wx_buffer_.get();
  double obj = 0;

  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = this->UpdateSingleExampleGradient(free_params, wx, example_idx,
                                                   subgrad, subgrad + pw_dim);
    obj += val;
    wx += kPW;
  }
  // normalize
  for (int i = 0; i < this->GetOracleParamsDim(); ++i) subgrad[i] /= nexamples;
  obj /= nexamples;

  return obj;
}

template <class Loss>
void VilmaOracle::PwVilmaRegularized<Loss>::ProjectData(const DenseVecD &aw,
                                                        Data *data,
                                                        double *wx_buffer,
                                                        const int kPW) {
  const int dim_x = data->GetDataDim();
  // create weight components
  vector<std::unique_ptr<DenseVecD>> w(kPW);
  for (int i = 0; i < kPW; ++i) {
    w[i].reset(new DenseVecD(dim_x, aw.data_ + dim_x * i));
  }

  const int nexamples = data->GetDataNumExamples();
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
double VilmaOracle::PwVilmaRegularized<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, double *const wx, const int example_idx,
    double *w_gradient, double *free_params_gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  const auto left_subproblem =
      PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup(
          wx, alpha_buffer_.get(), beta.data_, 0, gt_yl, gt_yl, kPW, &loss_);

  const auto right_subproblem =
      PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup(
          wx, alpha_buffer_.get(), beta.data_, gt_yr,
          data_->GetDataNumClasses() - 1, gt_yr, kPW, &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  if (w_gradient != nullptr) {
    // update gradient
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // TODO: make more efficient
    for (int y : {best_yl, best_yr}) {
      double *alpha = alpha_buffer_.get() + y * kPW;
      for (int k = 0; k < kPW; ++k) {
        double *grad = w_gradient + k * data_->GetDataDim();
        for (int i = 0; i < x->non_zero_; ++i) {
          grad[x->index_[i]] += alpha[k] * x->vals_[i];
        }
      }
    }

    for (int y : {gt_yl, gt_yr}) {
      double *alpha = alpha_buffer_.get() + y * kPW;
      for (int k = 0; k < kPW; ++k) {
        double *grad = w_gradient + k * data_->GetDataDim();
        for (int i = 0; i < x->non_zero_; ++i) {
          grad[x->index_[i]] -= alpha[k] * x->vals_[i];
        }
      }
    }
  }  // end w gradient update

  double psi = beta[gt_yl] + beta[gt_yr];
  for (int y : {gt_yl, gt_yr}) {
    double *alpha = alpha_buffer_.get() + y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  //  double *beta_grad = gradient->data_ + GetDataDim() * kPW;
  if (free_params_gradient != nullptr) {
    free_params_gradient[best_yl] += 1;
    free_params_gradient[best_yr] += 1;
    free_params_gradient[gt_yl] -= 1;
    free_params_gradient[gt_yr] -= 1;
  }

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template <class Loss>
std::pair<double, int>
VilmaOracle::PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup(
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
  assert(best_y != -1);
  return std::make_pair(best_cost, best_y);
}