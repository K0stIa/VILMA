/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef svor_imc_reg_h
#define svor_imc_reg_h

#include "ordinal_regression.h"

#include <memory>

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

class SvorImcReg : public OrdinalRegression {
 public:
  SvorImcReg() = delete;
  virtual ~SvorImcReg() = default;

  SvorImcReg(Data *data);

  using OrdinalRegression::risk;

  using OrdinalRegression::GetOracleParamsDim;
  using OrdinalRegression::SingleExampleBestLabelLookup;
  using OrdinalRegression::ProjectData;
  using OrdinalRegression::GetOracleData;
  using OrdinalRegression::Train;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &theta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using OrdinalRegression::dim;
  using OrdinalRegression::data_;
  using OrdinalRegression::wx_buffer_;
};

}  // namespace BmrmOracle

#endif /* svor_imc_reg_h */
