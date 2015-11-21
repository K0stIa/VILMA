/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "accpm_constrained_theta_tail_parameters_oracle.h"

using namespace VilmaAccpmOracle;

AccpmConstrainedThetaTailParametersOracle::
    AccpmConstrainedThetaTailParametersOracle(
        VilmaOracle::OrdinalRegression *ord_oracle, double *wx_buffer,
        const int dim, const int dw)
    : AccpmTailParametersOracle(ord_oracle, wx_buffer, dim, dw) {}

int AccpmConstrainedThetaTailParametersOracle::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kDim; ++i) {
    params[i] = y(i);
  }
  gradient.fill(0);

  // check for feasibility
  bool feasibly = true;
  for (int i = 0; i < kDim - 1; ++i) {
    if (params[i] > params[i + 1]) {
      feasibly = false;
    }
  }

  if (feasibly) {
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
  } else {
    // not feasibile point, so we do a feasibility cut
    // in OBOE if we have a linear constraint d^T * y <= d
    // d^T * (y - x) + d^T * x - d <= 0, where x is a query point.
    // Thus oracle has to return vector d and scalar value d^T * x - d

    // find wrongly ordered thresholds, e.g. theta_y > theta_{y+1}

    // https://github.com/lolow/oboe-fix/blob/master/doc/userguide/OBOE-UserGuide.pdf
    // section 3.1.2

    int idx = -1;
    for (int i = 0; i < kDim - 1; ++i) {
      if (params[i] > params[i + 1]) {
        idx = i;
        break;
      }
    }
    assert(idx != -1);
    for (int i = 0; i < kDim; ++i) {
      accpm_grad_vector_(i) = 0;
    }
    accpm_grad_vector_(idx + 1) = -1;
    accpm_grad_vector_(idx) = 1;

    // assigning scalar
    functionValue = params[idx] - params[idx + 1];

    // tell oracle it is  feasibility cut
    if (info != nullptr) {
      *info = 0;
    }
  }

  memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
         sizeof(double) * accpm_grad_vector_.size());

  return 0;
}