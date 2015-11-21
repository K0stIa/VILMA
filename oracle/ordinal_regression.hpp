/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef ordinal_regression_hpp
#define ordinal_regression_hpp

#include <vector>

#include "ordinal_regression.h"
#include "mord_regularized.h"

#include "pw_vilma_regularized.h"

namespace VilmaOracle {

template <class Loss>
double EvaluateModel(Data *data, double *params) {
  const int num_examples = data->GetDataNumExamples();
  const int dim_x = data->GetDataDim();
  Loss loss;
  int error = 0;
  DenseVecD theta(data->GetDataNumClasses() - 1, params + dim_x);
  for (int ex = 0; ex < num_examples; ++ex) {
    const Vilma::SparseVector<double> &x = *data->x->GetRow(ex);
    const int y = data->y->data_[ex];
    const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
    int pred_y = OrdinalRegression::SingleExampleBestLabelLookup(
        wx, theta, data->GetDataNumClasses());
    error += loss(y, pred_y);
  }

  return 1. * error / num_examples;
}

template <class Loss>
double EvaluateMordModel(Data *data, double *params) {
  const int num_examples = data->GetDataNumExamples();
  const int dim_x = data->GetDataDim();
  Loss loss;
  int error = 0;
  DenseVecD beta(data->GetDataNumClasses(), params + dim_x);
  for (int ex = 0; ex < num_examples; ++ex) {
    const Vilma::SparseVector<double> &x = *data->x->GetRow(ex);
    const int y = data->y->data_[ex];
    const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
    auto ret = MOrdRegularized<Loss>::SingleExampleBestLabelLookup(
        wx, beta, 0, data->GetDataNumClasses() - 1, -1, &loss);
    int pred_y = std::get<1>(ret);
    error += loss(y, pred_y);
  }

  return 1. * error / num_examples;
}

template <class Loss>
double EvaluatePwModel(Data *data, double *params,
                       const std::vector<int> &cut_labels) {
  const int num_examples = data->GetDataNumExamples();
  const int dim_x = data->GetDataDim();
  const int ny = data->GetDataNumClasses();
  Loss loss;
  int error = 0;

  const int kPW = (int)cut_labels.size();
  double *alpha_buffer = PwVilmaRegularized<Loss>::BuildAlphas(cut_labels, ny);

  std::vector<double> wx(dim_x * kPW);
  DenseVecD weights(dim_x * kPW, params);

  PwVilmaRegularized<Loss>::ProjectData(weights, data, &wx[0], kPW);

  const double *beta = params + dim_x * kPW;

  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    const int gt_y = data->y->operator[](example_idx);

    const auto subproblem_res =
        PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup(
            &wx[0] + example_idx * kPW, alpha_buffer, beta, 0, ny - 1, -1, kPW,
            nullptr);

    const int &best_y = std::get<1>(subproblem_res);

    error += loss(gt_y, best_y);
  }

  return 1. * error / num_examples;
}

}  // namespace VilmaOracle

#endif /* ordinal_regression_hpp */
