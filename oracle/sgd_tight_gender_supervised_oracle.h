/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__sgd_tight_gender_supervised_oracle__
#define __sparse_sgd__sgd_tight_gender_supervised_oracle__

#include "data.h"
#include "dense_vector.h"
#include "sparse_matrix.h"

#include <stdio.h>

class Data;

namespace VilmaOracle {

template <class Loss>
class TightGenderSupervisedOracle {
 public:
  typedef Vilma::DenseVector<double> DenseVecD;
  typedef Vilma::DenseVectorView<double> DenseVecDView;
  typedef Loss OracleLoss;

  TightGenderSupervisedOracle(const Data *const data);
  double UpdateGradient(const Vilma::DenseVector<double> &params,
                        const int example_idx,
                        Vilma::DenseVector<double> *gradient);

  int GetDataDim() { return data_->x->kCols; }
  int GetOracleParamsDim() { return (GetDataDim() + data_->ny) * data_->nz; }
  int GetDataNumExamples() { return data_->x->kRows; }
  const Data *const GetDataPtr() { return data_; }

  const Data *const data_;

 protected:
  template <class T, class V>
  void Convert2Beta(const T &theta, V &beta);

  std::tuple<double, int> best_y_lookup(const double *beta, const double wx,
                                        const int y, const double psi,
                                        const int from_label,
                                        const int to_label, const int ny);

 private:
  const int kNz;
  const Loss loss;
  // std::unique_ptr<DenseVecD> beta_ptr;
};
}

#include "sgd_tight_gender_supervised_oracle.hpp"

#endif /* defined(__sparse_sgd__sgd_tight_gender_supervised_oracle__) */
