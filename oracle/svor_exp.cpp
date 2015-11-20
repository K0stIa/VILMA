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

#include "tail_parameters_oracle.h"
#include "accpm_parameters_builder.h"
#include "accpm_constrained_theta_tail_parameters_oracle.h"

#include "svor_exp.h"

using namespace VilmaOracle;

SvorExp::SvorExp(Data *data) : SvorImc(data) {
  free_parameters_oracle_.ResetAccpmTailParametersOracle(
      new VilmaAccpmOracle::AccpmConstrainedThetaTailParametersOracle(
          this, wx_buffer_.get(), theta_.dim_));
}

double SvorExp::UpdateSingleExampleGradient(const DenseVecD &theta,
                                            const double wx,
                                            const int example_idx,
                                            double *w_gradient,
                                            double *free_params_gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  double obj = 0;
  int coef = 0;

  if (gt_yl > 0) {
    double val = 1.0 - wx + theta[gt_yl - 1];
    if (val > 0) {
      obj += val;
      --coef;
      if (free_params_gradient != nullptr) {
        free_params_gradient[gt_yl - 1] += 1;
      }
    }
  }

  if (gt_yr < data_->ny - 1) {
    double val = 1.0 + wx - theta[gt_yr];
    if (val > 0) {
      obj += val;
      ++coef;
      if (free_params_gradient != nullptr) {
        free_params_gradient[gt_yr] -= 1;
      }
    }
  }

  if (w_gradient != nullptr && coef != 0) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // TODO: make method for this
    for (int i = 0; i < x->non_zero_; ++i) {
      w_gradient[x->index_[i]] += coef * x->vals_[i];
    }
  }

  return obj;
}