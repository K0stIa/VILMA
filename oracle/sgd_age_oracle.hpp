//
//  sgd_age_oracle.cpp
//  sparse_sgd
//
//  Created by Kostia on 3/4/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "sgd_age_oracle.h"

namespace VilmaOracle {

template <class Loss>
AgeOracle<Loss>::AgeOracle(const Data *const data)
    : data_(data), loss_() {}

template <class Loss>
double AgeOracle<Loss>::UpdateGradient(
    const Vilma::DenseVector<double> &params, const int example_idx,
    Vilma::DenseVector<double> *gradient) {
  const Data &kData = *data_;
  // current example feature
  const Vilma::SparseVector<double> &x = *kData.x->GetRow(example_idx);

  // extract example labels
  const int yl = kData.yl->data_[example_idx];
  const int yr = kData.yr->data_[example_idx];
  const int z = kData.z->data_[example_idx];

  assert(0 <= z && z <= kData.nz);
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
  // feature dimension
  const int dim_x = kData.x->kCols;

  DenseVecD w(dim_x, params.data_);
  const double wx = x.dot<DenseVecD>(w);
  const double *beta = params.data_ + dim_x;

  auto left_subproblem = best_y_lookup(wx, beta, 0, yl, yl, &loss_);
  auto right_subproblem =
      best_y_lookup(wx, beta, yr, data_->ny - 1, yr, &loss_);

  const int best_yl = std::get<1>(left_subproblem);
  const int best_yr = std::get<1>(right_subproblem);
  // update gradient
  const int coef = best_yl + best_yr - yl - yr;

  for (int i = 0; i < x.non_zero_; ++i) {
    gradient->data_[x.index_[i]] += x.vals_[i] * coef;
  }

  gradient->data_[dim_x + best_yl] += 1;
  gradient->data_[dim_x + best_yr] += 1;
  gradient->data_[dim_x + yl] -= 1;
  gradient->data_[dim_x + yr] -= 1;

  const double psi = wx * (yl + yr) - beta[yl] - beta[yr];

  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;
}

template <class Loss>
std::tuple<double, int> AgeOracle<Loss>::best_y_lookup(
    const double wx, const double *beta, int from, int to, const int y,
    const Loss *const loss_ptr_) {
  double best_cost = 0;
  int best_y = -1;
  for (int l = from; l <= to; ++l) {
    const double cost =
        (loss_ptr_ != nullptr ? loss_ptr_->operator()(l, y) : 0) + wx * l +
        beta[l];
    if (best_y == -1 || best_cost < cost) {
      best_cost = cost;
      best_y = l;
    }
  }
  assert(best_y != -1);
  return std::make_tuple(best_cost, best_y);
}
}

//|-------------------------------------------------------------------------------------|
//|---------------------------------BetaAuxiliaryOracle---------------------------------|
//|-------------------------------------------------------------------------------------|

template <class Loss>
VilmaOracle::BetaAuxiliaryOracle<Loss>::BetaAuxiliaryOracle(
    const Data *const data, const DenseVecD &weights)
    : data_(data), loss_() {
  wx = new double[GetDataNumExamples()];
  const Data &kData = *data;
  for (int example_idx = 0; example_idx < GetDataNumExamples(); ++example_idx) {
    // current example feature
    const Vilma::SparseVector<double> &x = *kData.x->GetRow(example_idx);
    wx[example_idx] = x.dot<DenseVecD>(weights);
  }
}

template <class Loss>
VilmaOracle::BetaAuxiliaryOracle<Loss>::~BetaAuxiliaryOracle() {
  if (wx != nullptr) {
    delete[] wx;
  }
}

template <class Loss>
double VilmaOracle::BetaAuxiliaryOracle<Loss>::UpdateGradient(
    const Vilma::DenseVector<double> &params, const int example_idx,
    Vilma::DenseVector<double> *gradient) {
  const Data &kData = *data_;
  // extract example labels
  const int yl = kData.yl->data_[example_idx];
  const int yr = kData.yr->data_[example_idx];
  const int z = kData.z->data_[example_idx];

  assert(0 <= z && z <= 1);
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);

  // precomputed already
  double wx = this->wx[example_idx];

  const double *beta = params.data_;

  auto left_subproblem =
      AgeOracle<Loss>::best_y_lookup(wx, beta, 0, yl, yl, &loss_);
  auto right_subproblem =
      AgeOracle<Loss>::best_y_lookup(wx, beta, yr, data_->ny - 1, yr, &loss_);
  const int best_yl = std::get<1>(left_subproblem);
  const int best_yr = std::get<1>(right_subproblem);

  // update gradient

  gradient->data_[best_yl] += 1;
  gradient->data_[best_yr] += 1;
  gradient->data_[yl] -= 1;
  gradient->data_[yr] -= 1;

  //  const double psi = wx * yl + beta[yl];
  //  return std::get<0>(subproblem) - psi;
  const int coef = best_yl + best_yr - yl - yr;
  return loss_(best_yl, yl) + loss_(best_yr, yr) + wx * coef + beta[best_yl] +
         beta[best_yr] - beta[yl] - beta[yr];
}

template <class Loss>
VilmaOracle::AgeTemplatedOracle<Loss>::AgeTemplatedOracle(const Data *const data)
    : data_(data), loss_() {}

template <class Loss>
double VilmaOracle::AgeTemplatedOracle<Loss>::UpdateGradient(
    const Vilma::DenseVector<double> &params, const int example_idx,
    Vilma::DenseVector<double> *gradient) {
  // get data reference
  const Data &kData = *data_;
  // current example feature
  const Vilma::SparseVector<double> &x = *kData.x->GetRow(example_idx);

  // extract example labels
  const int yl = kData.yl->data_[example_idx];
  const int yr = kData.yr->data_[example_idx];
  const int z = kData.z->data_[example_idx];

  assert(0 <= z && z <= 1);
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
  // feature dimension
  const int dim_x = kData.x->kCols;

  DenseVecD w(dim_x, params.data_);
  double wx = x.dot<DenseVecD>(w);
  //  for (int i = 0; i < x.non_zero_; ++i) {
  //    assert(x.index_[i] < dim_x);
  //    assert(0 <= x.index_[i]);
  //    wx += params.data_[x.index_[i]];
  //  }

  const double *beta = params.data_ + dim_x;

  //  auto left_subproblem = best_y_lookup(wx, beta, 0, yl, yl, );
  //  auto right_subproblem = best_y_lookup(wx, beta, yr, data_->ny - 1, yr,
  //  );
  //
  //  // update gradient
  //  //  DenseVecD(dim_x, gradient->data_)
  //  //      .add_sparse(x, std::get<1>(left_subproblem) +
  //  //                         std::get<1>(right_subproblem) - yl - yr);
  //  const int coef =
  //      std::get<1>(left_subproblem) + std::get<1>(right_subproblem) - yl -
  //      yr;
  //
  //  for (int i = 0; i < x.non_zero_; ++i) {
  //    gradient->data_[x.index_[i]] += coef;
  //  }
  //
  //  for (int y : {std::get<1>(left_subproblem),
  //  std::get<1>(right_subproblem)}) {
  //    gradient->data_[dim_x + y] += 1;
  //  }
  //
  //  for (int y : {yl, yr}) {
  //    gradient->data_[dim_x + y] -= 1;
  //  }
  //
  //  const double psi = wx * (yl + yr) + beta[yl] + beta[yr];
  //  return std::get<0>(left_subproblem) + std::get<0>(right_subproblem) - psi;

  auto left_subproblem = best_y_lookup(wx, beta, 0, yl, yl, &loss_);
  auto right_subproblem =
      best_y_lookup(wx, beta, yr, data_->ny - 1, yr, &loss_);
  const int best_yl = std::get<1>(left_subproblem);
  const int best_yr = std::get<1>(right_subproblem);
  // update gradient
  const int coef = best_yl + best_yr - yl - yr;

  for (int i = 0; i < x.non_zero_; ++i) {
    gradient->data_[x.index_[i]] += coef;
  }

  gradient->data_[dim_x + best_yl] += 1;
  gradient->data_[dim_x + best_yr] += 1;
  gradient->data_[dim_x + yl] -= 1;
  gradient->data_[dim_x + yr] -= 1;

  //  const double psi = wx * yl + beta[yl];
  //  return std::get<0>(subproblem) - psi;
  return loss_(best_yl, yl) + loss_(best_yr, yr) + wx * coef + beta[best_yl] +
         beta[best_yr] - beta[yl] - beta[yr];
}