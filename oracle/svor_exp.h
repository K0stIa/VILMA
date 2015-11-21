/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef svor_exp_
#define svor_exp_

#include "svor_imc.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

class SvorExp : public SvorImc {
 public:
  SvorExp() = delete;
  virtual ~SvorExp() = default;

  SvorExp(Data *data);

  using SvorImc::GetOracleParamsDim;
  using SvorImc::SingleExampleBestLabelLookup;
  using SvorImc::ProjectData;
  using SvorImc::risk;
  using SvorImc::Train;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &theta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using SvorImc::dim;
  using SvorImc::data_;
  using SvorImc::wx_buffer_;
  using SvorImc::theta_;
  using SvorImc::free_parameters_oracle_;
};
}  // namespace VilmaOracle

#endif /* svor_exp_ */
