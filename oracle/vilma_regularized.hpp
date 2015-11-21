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

template <class Loss>
VilmaOracle::VilmaRegularized<Loss>::VilmaRegularized(Data *data)
    : MOrdRegularized<Loss>(data) {}

template <class Loss>
double VilmaOracle::VilmaRegularized<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, double *const wx, const int example_idx,
    double *w_gradient, double *free_params_gradient) {
  // extract example labels
  const int gt_y = data_->y->data_[example_idx];

  const auto subproblem = SingleExampleBestLabelLookup(
      *wx, beta, 0, data_->GetDataNumClasses() - 1, gt_y, &loss_);

  const int &best_y = std::get<1>(subproblem);

  // find gradient coeficient
  const int coef = best_y - gt_y;

  // update w gradiet if it exists
  if (w_gradient != nullptr && coef != 0) {
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    for (int i = 0; i < x->non_zero_; ++i) {
      w_gradient[x->index_[i]] += coef * x->vals_[i];
    }
  }

  // update beta gradient if it exists
  if (free_params_gradient != nullptr) {
    free_params_gradient[best_y] += 1;
    free_params_gradient[gt_y] -= 1;
  }

  const double psi = *wx * gt_y + beta[gt_y];

  return std::get<0>(subproblem) - psi;
}