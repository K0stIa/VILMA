/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "data.h"
#include "accpm_parameters_builder.h"
#include "accpm_tail_parameters_oracle.h"

#include "svor_imc.h"

using namespace VilmaOracle;

SvorImc::SvorImc(Data *data)
    : SvorImcReg(data),
      theta_(data->GetDataNumClasses() - 1),
      free_parameters_oracle_(
          new VilmaAccpmOracle::AccpmTailParametersOracle(
              this, wx_buffer_.get(), data->GetDataNumClasses() - 1),
          new VilmaAccpmOracle::VilmaAccpmParametersBuilder(
              data->GetDataNumClasses() - 1)) {
  dim = data->GetDataDim();
}

double SvorImc::risk(const double *weights, double *subgrad) {
  const int nexamples = data_->GetDataNumExamples();
  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);
  gradient.fill(0);

  ProjectData(params, data_, wx_buffer_.get());
  // train theta
  std::vector<double> opt_theta = free_parameters_oracle_.Optimize();
  for (int i = 0; i < (int)opt_theta.size(); ++i) {
    theta_[i] = opt_theta[i];
  }

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = this->UpdateSingleExampleGradient(
        theta_, wx_buffer_[example_idx], example_idx, gradient.data_, nullptr);
    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

std::vector<double> SvorImc::Train() {
  std::vector<double> opt_w = BMRM_Solver::learn();
  DenseVecD params(dim, &opt_w[0]);

  ProjectData(params, data_, wx_buffer_.get());

  std::vector<double> opt_theta = free_parameters_oracle_.Optimize();

  opt_w.insert(opt_w.end(), opt_theta.begin(), opt_theta.end());

  return opt_w;
}