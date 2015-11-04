/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __vilma__model_evaluator__
#define __vilma__model_evaluator__

#include "dense_vector.h"

#include <stdio.h>

class Data;

template <class Loss>
class ModelEvaluator {
 public:
  template <class T>
  using DenseVec = Vilma::DenseVector<T>;

  template <class T>
  using DenseVecView = Vilma::DenseVectorView<T>;

  typedef DenseVec<double> DenseVecD;
  typedef DenseVecView<double> DenseVecDView;
  typedef DenseVec<int> DenseVecInt;

  ModelEvaluator(const Data *);

  double CalcError(const DenseVecD &);

 protected:
  void predict(const DenseVecD &, const Data *, DenseVecInt *);

  template <class T, class V>
  void Convert2Beta(const T &theta, V &beta);
  std::tuple<double, int> best_y_lookup(const double *beta, const double wx,
                                        const int ny);

 private:
  const Data *const data_;
};

#include "model_evaluator.hpp"

#endif /* defined(__vilma__model_evaluator__) */
