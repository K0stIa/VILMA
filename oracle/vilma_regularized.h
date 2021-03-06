/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Written (W) 2015 Kostiantyn Antoniuk
* Copyright (C) 2015 Kostiantyn Antoniuk
*/

#ifndef __VilmaRegularized__
#define __VilmaRegularized__

#include "ordinal_regression.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class VilmaRegularized : public OrdinalRegression {
 public:
  VilmaRegularized() = delete;

  VilmaRegularized(Data *data);
  virtual ~VilmaRegularized() = default;

  using OrdinalRegression::GetOracleParamsDim;
  using OrdinalRegression::GetOracleData;
  using OrdinalRegression::ProjectData;
  using OrdinalRegression::Train;
  using OrdinalRegression::risk;

  static std::tuple<double, int> SingleExampleBestLabelLookup(
      const double wx, const DenseVecD &beta, int from, int to, const int gt_y,
      const Loss *const loss_ptr_);

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using OrdinalRegression::dim;
  Loss loss_;
  // Oracle is never an owner of Data
  using OrdinalRegression::data_;
  // buffer to store results <w,x> for all x
  using OrdinalRegression::wx_buffer_;
};
}

#include "vilma_regularized.hpp"

#endif /* defined(__VilmaRegularized__) */
