/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef ordinal_regression_h
#define ordinal_regression_h

#include <stdio.h>
#include <vector>

#include "../bmrm/bmrm_solver.h"
#include "dense_vector.h"
#include "data.h"

namespace VilmaOracle {
typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

class OrdinalRegression : public BMRM_Solver {
 public:
  OrdinalRegression() = delete;
  virtual ~OrdinalRegression() = default;

  OrdinalRegression(Data *data);

  int GetOracleParamsDim();
  Data *GetOracleData();
  double *GetWxBuffer();

  virtual double risk(const double *weights, double *subgrad);

  static void ProjectData(const DenseVecD &w, Data *data, double *wx_buffer);

  static int SingleExampleBestLabelLookup(const double wx,
                                          const DenseVecD &theta,
                                          const int data_num_classes);

  std::vector<double> Train();

  virtual double UpdateSingleExampleGradient(const DenseVecD &theta,
                                             const double wx,
                                             const int example_idx,
                                             double *w_gradient,
                                             double *free_params_gradient) = 0;

 protected:
  using BMRM_Solver::dim;

  // Oracle is never an owner of Data
  Data *data_ = nullptr;
  // buffer to store results <w,x> for all x
  std::unique_ptr<double[]> wx_buffer_;
};

}  // namespace VilmaOracle

#include "ordinal_regression.hpp"

#endif /* ordinal_regression_h */
