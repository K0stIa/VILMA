/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "model_evaluator.h"
#include "data.h"

#include <assert.h>
#include <tuple>

template <class Loss>
ModelEvaluator<Loss>::ModelEvaluator(const Data *const data)
    : data_(data) {}

template <class Loss>
template <class T, class V>
void ModelEvaluator<Loss>::Convert2Beta(const T &theta, V &beta) {
  assert(theta.dim_ + 1 == beta.dim_);
  beta.data_[0] = 0;
  //  for (int i = 0; i < theta.dim_; ++i) {
  //    beta.data_[i + 1] = beta.data_[i] - theta.data_[i];
  //  }
  for (int i = 0; i < theta.dim_; ++i) {
    beta.data_[i + 1] = theta.data_[i];
  }
}

template <class Loss>
void ModelEvaluator<Loss>::predict(const DenseVecD &model, const Data *data,
                                   DenseVecInt *labels) {
  const int num_examples = data->x->kRows;

  assert(labels != nullptr);

  // feature dimension
  const int dim_x = data->x->kCols;

  // conver theta to beta
  const double *beta_base = model.data_ + dim_x * data->nz;

  for (int example = 0; example < num_examples; ++example) {
    // current example feature
    const Vilma::SparseVector<double> *x = data->x->GetRow(example);

    // make inferencer
    int best_y = -1, best_z = -1;
    double best_cost = 0;

    for (int k = 0; k < data->nz; ++k) {
      DenseVecDView w(model, dim_x * k, dim_x * (k + 1));
      // compute dot prodact of template weifghts per eaach gender with features
      const double wx = x->dot<DenseVecDView>(w);

      std::tuple<double, int> res =
          best_y_lookup(beta_base + data->ny * k, wx, data->ny);
      const double cost = std::get<0>(res);
      if (best_y == -1 || best_cost < cost) {
        best_cost = cost;
        best_y = std::get<1>(res);
        best_z = k;
      }
    }
    // store best label
    labels->data_[example] = best_y;
  }
}

template <class Loss>
double ModelEvaluator<Loss>::CalcError(const DenseVecD &model) {
  const int num_examples = data_->x->kRows;
  DenseVecInt labels(num_examples);

  predict(model, data_, &labels);

  int error = 0;
  for (int i = 0; i < labels.dim_; ++i) {
    const int &gt = data_->y->data_[i];
    const int &y = labels.data_[i];
    error += gt > y ? gt - y : y - gt;
  }
  return static_cast<double>(error) / labels.dim_;
}

template <class Loss>
std::tuple<double, int> ModelEvaluator<Loss>::best_y_lookup(const double *beta,
                                                            const double wx,
                                                            const int ny) {
  double best_cost = 0;
  int best_y = -1;
  for (int l = 0; l < ny; ++l) {
    double cost = wx * l + beta[l];
    if (best_y == -1 || best_cost < cost) {
      best_cost = cost;
      best_y = l;
    }
  }
  return std::make_tuple(best_cost, best_y);
}
