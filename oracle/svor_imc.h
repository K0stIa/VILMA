/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef svor_imc_
#define svor_imc_

#include "svor_imc_reg.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

class SvorImc : public SvorImcReg {
 public:
  SvorImc() = delete;
  virtual ~SvorImc() = default;

  SvorImc(Data *data);

  virtual double risk(const double *weights, double *subgrad) override;

  std::vector<double> Train();

  using SvorImcReg::GetOracleParamsDim;
  using SvorImcReg::SingleExampleBestLabelLookup;
  using SvorImcReg::ProjectData;
  using SvorImcReg::UpdateSingleExampleGradient;

 protected:
  using BMRM_Solver::dim;
  using SvorImcReg::data_;
  using SvorImcReg::wx_buffer_;
  DenseVecD theta_;
};
}  // namespace VilmaOracle

#endif /* svor_imc_ */
