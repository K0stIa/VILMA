/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "svor_imc_reg.h"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

using namespace VilmaOracle;

SvorImcReg::SvorImcReg(Data *data) : OrdinalRegression(data) {}

double SvorImcReg::UpdateSingleExampleGradient(const DenseVecD &theta,
                                               double *const wx,
                                               const int example_idx,
                                               double *w_gradient,
                                               double *free_params_gradient) {
  // extract example labels
  const int gt_yl = data_->yl->operator[](example_idx);
  const int gt_yr = data_->yr->operator[](example_idx);

  double obj = 0;
  int total_sgn = 0;

  for (int y = 0; y < data_->GetDataNumClasses() - 1; ++y) {
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
    double score = 1.0 - (*wx - theta[y]) * sgn;
    if (score > 0) {
      obj += score;
      if (free_params_gradient != nullptr) {
        free_params_gradient[y] += sgn;
      }
      total_sgn += sgn;
    }
  }

  if (w_gradient != nullptr && total_sgn != 0) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // update data cmponent of gradient
    for (int i = 0; i < x->non_zero_; ++i) {
      w_gradient[x->index_[i]] -= total_sgn * x->vals_[i];
    }
  }

  return obj;
}