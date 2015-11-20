//
//  mord_regularized.cpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#include "loss.h"

#include "mord_regularized.h"

using namespace VilmaOracle;

template <class Loss>
MOrdRegularized<Loss>::MOrdRegularized(Data *data)
    : OrdinalRegression(data) {
  dim = data->GetDataDim() + data->GetDataNumClasses();
}

template <class Loss>
std::tuple<double, int> MOrdRegularized<Loss>::SingleExampleBestLabelLookup(
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
  return std::make_tuple(best_cost, best_y);
}

template <class Loss>
double MOrdRegularized<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &beta, const double wx, const int example_idx,
    double *w_gradient, double *free_params_gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  const auto left_subproblem =
      SingleExampleBestLabelLookup(wx, beta, 0, gt_yl, gt_yl, &loss_);

  const auto right_subproblem = SingleExampleBestLabelLookup(
      wx, beta, gt_yr, data_->GetDataNumClasses() - 1, gt_yr, &loss_);

  const int &best_yl = std::get<1>(left_subproblem);
  const int &best_yr = std::get<1>(right_subproblem);

  // find gradient coeficient
  const int coef = best_yl + best_yr - gt_yl - gt_yr;

  // update w gradiet if it exists
  if (w_gradient != nullptr && coef != 0) {
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    for (int i = 0; i < x->non_zero_; ++i) {
      w_gradient[x->index_[i]] += coef * x->vals_[i];
    }
  }

  // update beta gradient if it exists
  if (free_params_gradient != nullptr) {
    free_params_gradient[best_yl] += 1;
    free_params_gradient[best_yr] += 1;
    free_params_gradient[gt_yl] -= 1;
    free_params_gradient[gt_yr] -= 1;
  }

  const double psi = wx * (gt_yl + gt_yr) + beta[gt_yl] + beta[gt_yr];

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template class VilmaOracle::MOrdRegularized<Vilma::MAELoss>;
