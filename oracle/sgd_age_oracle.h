/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__sgd_age_oracle__
#define __sparse_sgd__sgd_age_oracle__

#include "data.h"
#include "dense_vector.h"
#include "sparse_matrix.h"

#include <stdio.h>

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class AgeOracle {
 public:
  AgeOracle(const Data *const data);
  double UpdateGradient(const Vilma::DenseVector<double> &params,
                        const int example_idx,
                        Vilma::DenseVector<double> *gradient);

  int GetDataDim() { return data_->x->kCols; }
  int GetOracleParamsDim() { return GetDataDim() + data_->ny; }
  int GetDataNumExamples() { return data_->x->kRows; }
  const Data *const GetDataPtr() { return data_; }

  const Data *const data_;

  double EvaluateModel(Data *data, double *params) {
    const Data &kData = *data;
    const int num_examples = data->x->kRows;
    const int dim_x = GetDataDim();
    int error = 0;
    const double *beta = params + dim_x;
    for (int ex = 0; ex < num_examples; ++ex) {
      const Vilma::SparseVector<double> &x = *kData.x->GetRow(ex);
      const int y = kData.y->data_[ex];
      const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
      std::tuple<double, int> res =
          best_y_lookup(wx, beta, 0, kData.ny - 1, y, nullptr);
      error += abs(y - std::get<1>(res));
    }

    return 1. * error / num_examples;
  }

  static std::tuple<double, int> best_y_lookup(const double wx,
                                               const double *beta, int from,
                                               int to, const int y,
                                               const Loss *const loss_ptr_);

 private:
  typedef Loss OracleLoss;

  const Loss loss_;
};

template <class Loss>
class BetaAuxiliaryOracle {
 public:
  const Data *data_;

  int GetDataDim() { return 1; }
  int GetOracleParamsDim() { return data_->ny; }
  int GetDataNumExamples() { return data_->x->kRows; }
  const Data *const GetDataPtr() { return data_; }

  BetaAuxiliaryOracle(const Data *const data, const DenseVecD &weights);
  ~BetaAuxiliaryOracle();

  double UpdateGradient(const Vilma::DenseVector<double> &params,
                        const int example_idx,
                        Vilma::DenseVector<double> *gradient);

 private:
  typedef Loss OracleLoss;

  const Loss loss_;
  double *wx;
};

template <class Loss>
class AgeTemplatedOracle {
 public:
  AgeTemplatedOracle(const Data *const data);
  double UpdateGradient(const Vilma::DenseVector<double> &params,
                        const int example_idx,
                        Vilma::DenseVector<double> *gradient);

  int GetDataDim() { return data_->x->kCols; }
  int GetOracleParamsDim() { return GetDataDim(); }
  int GetDataNumExamples() { return data_->x->kRows; }
  const Data *const GetDataPtr() { return data_; }

  const Data *const data_;

 private:
  Loss loss_;
};
}

#include "sgd_age_oracle.hpp"

#endif /* defined(__sparse_sgd__sgd_age_oracle__) */
