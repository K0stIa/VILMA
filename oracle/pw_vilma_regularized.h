/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __PwVilmaRegularized__
#define __PwVilmaRegularized__

#include <vector>

#include "sparse_vector.h"
#include "ordinal_regression.h"

namespace VilmaOracle {

template <class Loss>
class PwVilmaRegularized : public OrdinalRegression {
 public:
  PwVilmaRegularized() = delete;

  PwVilmaRegularized(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwVilmaRegularized() = default;

  using OrdinalRegression::GetOracleData;
  using OrdinalRegression::Train;

  int GetOracleParamsDim();
  virtual int GetFreeParamsDim() override;

  virtual double risk(const double *weights, double *subgrad) override;

  static std::pair<double, int> SingleExampleBestLabelLookup(
      const double *wx, const double *alpha, const double *beta, int from,
      int to, const int gt_y, const int kPW, const Loss *const loss_ptr_);

  static void ProjectData(const DenseVecD &aw, Data *data, double *wx_buffer,
                          const int kPW);
  static double *BuildAlphas(const std::vector<int> &cut_labels, const int ny);

  const int kPW;

 protected:
  // store cut labels coef
  std::unique_ptr<double[]> alpha_buffer_;

  const std::vector<int> cut_labels;

  using OrdinalRegression::dim;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

  Loss loss_;
  // Oracle is never an owner of Data
  using OrdinalRegression::data_;
  // buffer to store results <w,x> for all x
  using OrdinalRegression::wx_buffer_;
};
}

#include "pw_vilma_regularized.hpp"

#endif /* defined(__PwVilmaRegularized__) */
