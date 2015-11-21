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
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>
#include <memory>

#include "pw_mord_regularized.h"

using namespace VilmaOracle;

template <class Loss>
PwMOrdRegularized<Loss>::PwMOrdRegularized(Data *data,
                                           const std::vector<int> &cut_labels)
    : PwVilmaRegularized<Loss>(data, cut_labels) {}

template <class Loss>
double PwMOrdRegularized<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, const double *wx, const int example_idx,
    double *w_gradient, double *free_params_gradient) {
  //
  const int gt_y = data_->y->data_[example_idx];
  auto subproblem = SingleExampleBestLabelLookup(
      wx, alpha_buffer_.get(), beta.data_, 0, data_->GetDataNumClasses() - 1,
      gt_y, kPW, &loss_);

  const int best_y = std::get<1>(subproblem);

  // update gradient
  if (w_gradient != nullptr) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // TODO: make more efficient
    {
      double *alpha = alpha_buffer_.get() + best_y * kPW;
      for (int k = 0; k < kPW; ++k) {
        double *grad = w_gradient + k * data_->GetDataDim();
        for (int i = 0; i < x->non_zero_; ++i) {
          grad[x->index_[i]] += alpha[k] * x->vals_[i];
        }
      }
    }

    {
      double *alpha = alpha_buffer_.get() + gt_y * kPW;
      for (int k = 0; k < kPW; ++k) {
        double *grad = w_gradient + k * data_->GetDataDim();
        for (int i = 0; i < x->non_zero_; ++i) {
          grad[x->index_[i]] -= alpha[k] * x->vals_[i];
        }
      }
    }
  }

  if (free_params_gradient != nullptr) {
    free_params_gradient[best_y] += 1;
    free_params_gradient[gt_y] -= 1;
  }

  double psi = beta[gt_y];
  {
    double *alpha = alpha_buffer_.get() + gt_y * kPW;
    // TODO: optimize
    for (int k = 0; k < kPW; ++k) {
      psi += alpha[k] * wx[k];
    }
  }

  return std::get<0>(subproblem) - psi;
}

template class VilmaOracle::PwMOrdRegularized<Vilma::MAELoss>;