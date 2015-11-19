/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef svor_imc_hpp
#define svor_imc_hpp

#include "svor_imc_reg.hpp"
#include "Oracle.h"

class Data;

namespace BmrmOracle {
template <class Loss>
class SvorImc : public SvorImcReg<Loss> {
 public:
  SvorImc() = delete;
  virtual ~SvorImc() = default;

  SvorImc(Data *data);

  using SvorImcReg<Loss>::GetDataDim;
  int GetOracleParamsDim();
  using SvorImcReg<Loss>::GetDataNumExamples;
  using SvorImcReg<Loss>::GetDataNumAgeClasses;
  using SvorImcReg<Loss>::ProjectData;
  using SvorImcReg<Loss>::EvaluateModel;

  virtual double risk(const double *weights, double *subgrad);

  static void TrainTheta(DenseVecD *theta, Data *data, double *wx_buffer);

 protected:
  double UpdateSingleExampleGradient(const DenseVecD &theta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

  using BMRM_Solver::dim;
  using SvorImcReg<Loss>::wx_buffer_;
  using SvorImcReg<Loss>::data_;

 private:
  // buffer to store thetas
  DenseVecD theta_;
};

template <class Loss>
class SvorImcAuxiliaryThetaAccpmOracle : public Accpm::OracleFunction {
 public:
  SvorImcAuxiliaryThetaAccpmOracle(Data *data, double *wx);
  ~SvorImcAuxiliaryThetaAccpmOracle();
  void UpdateWX(const double *wx);

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info);

 protected:
  double UpdateSingleExampleGradient(const DenseVecD &theta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

  const int kNumAgeClasses;
  DenseVecD params_, gradient_;
  Data *data_;
  Accpm::AccpmVector accpm_grad_vector_;
  double *wx_buffer_;
  Loss loss_;
};
}

#endif /* svor_imc_hpp */
