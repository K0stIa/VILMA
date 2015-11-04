/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "sgd_tight_gender_supervised_oracle.h"

#include <vector>
#include <iostream>

namespace VilmaOracle {

template <class Loss>
TightGenderSupervisedOracle<Loss>::TightGenderSupervisedOracle(
    const Data *const data)
    : data_(data), kNz(data->nz), loss() {
  // beta_ptr.reset(new DenseVecD(data->nz * data->ny));
}

template <class Loss>
double TightGenderSupervisedOracle<Loss>::UpdateGradient(
    const Vilma::DenseVector<double> &params, const int example_idx,
    Vilma::DenseVector<double> *gradient) {
  const Data &kData = *data_;
  // current example feature
  const Vilma::SparseVector<double> &x = *kData.x->GetRow(example_idx);

  // extract example labels
  const int yl = kData.y->data_[example_idx];
  const int yr = kData.y->data_[example_idx];
  const int z = kData.z->data_[example_idx];

  assert(0 <= z && z <= 1);
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
  // feature dimension
  const int dim_x = x.GetDim();

  assert(kNz == 2);

  double *wx = new double[kNz];
  // compute dot prodact of template weifghts per eaach gender with features
  for (int i = 0; i < kNz; ++i) {
    DenseVecDView w(params, dim_x * i, dim_x * (i + 1));
    wx[i] = x.dot<DenseVecDView>(w);
  }

  double *beta_base = params.data_ + dim_x * kData.nz;

  // conver theta to beta
  //  for (int k = 0; k < kData.nz; ++k) {
  //    const int shift = dim_x * kData.nz;
  //    DenseVecDView theta(params, shift + k * (kData.ny - 1),
  //                        shift + (k + 1) * (kData.ny - 1));
  //    DenseVecDView beta(*beta_ptr, k * kData.ny, (k + 1) * kData.ny);
  //    Convert2Beta<DenseVecDView, DenseVecDView>(theta, beta);
  //  }

  // compute Psis
  const double psi_l = wx[z] * yl + beta_base[kData.ny * z + yl];
  const double psi_r = wx[z] * yr + beta_base[kData.ny * z + yr];

  // make inference
  int best_yl = -1, best_yr = -1, best_z = -1;
  double best_cost = 0;

  for (int k = 0; k < kData.nz; ++k) {
    // DenseVecDView beta(*beta_ptr, k * kData.ny, (k + 1) * kData.ny);

    std::tuple<double, int> res_left = best_y_lookup(
        beta_base + k * kData.ny, wx[k], yl, psi_l, 0, yl, kData.ny);

    std::tuple<double, int> res_right = best_y_lookup(
        beta_base + k * kData.ny, wx[k], yr, psi_r, yr, kData.ny - 1, kData.ny);

    const double cost = std::get<0>(res_left) + std::get<0>(res_right);

    if (best_yl == -1 || best_cost < cost) {
      best_cost = cost;
      best_yl = std::get<1>(res_left);
      best_yr = std::get<1>(res_right);
      best_z = k;
    }
  }

  // update gradient, i.e. directly write to params
  // first procces W
  for (std::pair<int, int> p : std::vector<std::pair<int, int> >{
           {best_yl + best_yr, best_z}, {-yl - yr, z}}) {
    const int &y = p.first;
    const int &g = p.second;
    DenseVecDView grad(*gradient, dim_x * g, dim_x * (g + 1));
    grad.add_sparse(x, 1. * y);
  }
  // update theta gradient
  double *beta_grad = gradient->data_ + dim_x * kNz;
  {
    for (int y : {best_yl, best_yr}) {
      beta_grad[y + best_z * kData.ny] += 1.0;
    }
    for (int y : {yl, yr}) {
      beta_grad[y + z * kData.ny] -= 1.0;
    }
  }
  delete[] wx;

  return best_cost;
}

template <class Loss>
std::tuple<double, int> TightGenderSupervisedOracle<Loss>::best_y_lookup(
    const double *beta, const double wx, const int y, const double psi,
    const int from_label, const int to_label, const int ny) {
  double best_cost = 0;
  int best_y = -1;
  for (int l = from_label; l <= to_label; ++l) {
    double cost = loss(l, y) + wx * l + beta[l] - psi;
    if (best_y == -1 || best_cost < cost) {
      best_cost = cost;
      best_y = l;
    }
  }
  return std::make_tuple(best_cost, best_y);
}

template <class Loss>
template <class T, class V>
void TightGenderSupervisedOracle<Loss>::Convert2Beta(const T &theta, V &beta) {
  assert(theta.dim_ + 1 == beta.dim_);
  beta.data_[0] = 0;
  //  for (int i = 0; i < theta.dim_; ++i) {
  //    beta.data_[i + 1] = beta.data_[i] - theta.data_[i];
  //  }
  for (int i = 0; i < theta.dim_; ++i) {
    beta.data_[i + 1] = theta.data_[i];
  }
}
}