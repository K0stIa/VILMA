/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "pw_vilma.h"

template <class Loss>
VilmaOracle::PwVilma<Loss>::PwVilma(Data *data,
                                    const std::vector<int> &cut_labels)
    : PwVilmaRegularized<Loss>(data, cut_labels),
      beta_(data->GetDataNumClasses()),
      free_parameters_oracle_(
          new VilmaAccpmOracle::AccpmTailParametersOracle(
              this, wx_buffer_.get(), data->GetDataNumClasses(),
              (int)cut_labels.size()),
          new VilmaAccpmOracle::VilmaAccpmParametersBuilder(
              data->GetDataNumClasses())) {}

template <class Loss>
int VilmaOracle::PwVilma<Loss>::GetFreeParamsDim() {
  return 0;
}

template <class Loss>
double VilmaOracle::PwVilma<Loss>::risk(const double *weights,
                                        double *subgrad) {
  std::fill(subgrad, subgrad + this->GetOracleParamsDim(), 0);

  const int nexamples = data_->GetDataNumExamples();

  DenseVecD w(this->GetOracleParamsDim(), const_cast<double *>(weights));
  ProjectData(w, data_, wx_buffer_.get(), kPW);

  std::vector<double> opt_beta = free_parameters_oracle_.Optimize();
  for (int i = 0; i < (int)opt_beta.size(); ++i) {
    beta_[i] = opt_beta[i];
  }

  double *wx = wx_buffer_.get();
  double obj = 0;

  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = this->UpdateSingleExampleGradient(beta_, wx, example_idx,
                                                   subgrad, nullptr);
    obj += val;
    wx += kPW;
  }
  // normalize
  for (int i = 0; i < this->GetOracleParamsDim(); ++i) subgrad[i] /= nexamples;
  obj /= nexamples;

  return obj;
}

template <class Loss>
std::vector<double> VilmaOracle::PwVilma<Loss>::Train() {
  std::vector<double> opt_w = BMRM_Solver::learn();
  DenseVecD params(BMRM_Solver::dim, &opt_w[0]);

  ProjectData(params, data_, wx_buffer_.get(), this->kPW);

  std::vector<double> opt_beta = free_parameters_oracle_.Optimize();

  opt_w.insert(opt_w.end(), opt_beta.begin(), opt_beta.end());

  return opt_w;
}