/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "QpGenerator.h"

#include "accpm_tail_parameters_oracle.h"

using namespace VilmaAccpmOracle;

int AccpmTailParametersOracle::eval(const Accpm::AccpmVector &y,
                                    Accpm::AccpmVector &functionValue,
                                    Accpm::AccpmGenMatrix &subGradients,
                                    Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kDim; ++i) {
    params[i] = y(i);
  }
  gradient.fill(0);

  const int nexamples = ord_oracle->GetOracleData()->GetDataNumExamples();
  double obj = 0;

  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = ord_oracle->UpdateSingleExampleGradient(
        params, wx_buffer_ + example_idx * kDw, example_idx, nullptr,
        gradient.data_);

    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  functionValue = obj;

  for (int i = 0; i < kDim; ++i) {
    accpm_grad_vector_(i) = gradient[i];
  }

  if (info != nullptr) {
    *info = 1;
  }

  memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
         sizeof(double) * accpm_grad_vector_.size());

  return 0;
}