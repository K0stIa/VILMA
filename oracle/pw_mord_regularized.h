/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__pw_mord_no_beta_bmrm_oracle__
#define __sparse_sgd__pw_mord_no_beta_bmrm_oracle__

#include <stdio.h>
#include <vector>

#include "dense_vector.h"

#include "pw_vilma_regularized.h"

struct Data;

namespace VilmaOracle {

template <class Loss>
class PwMOrdRegularized : public PwVilmaRegularized<Loss> {
 public:
  PwMOrdRegularized() = delete;

  PwMOrdRegularized(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwMOrdRegularized() = default;

  using PwVilmaRegularized<Loss>::GetOracleData;
  using PwVilmaRegularized<Loss>::GetOracleParamsDim;
  using PwVilmaRegularized<Loss>::GetFreeParamsDim;
  using PwVilmaRegularized<Loss>::ProjectData;
  using PwVilmaRegularized<Loss>::BuildAlphas;
  using PwVilmaRegularized<Loss>::risk;
  using PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup;
  using PwVilmaRegularized<Loss>::Train;

 protected:
  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override {
    throw;
  }

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, const double *wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

  using PwVilmaRegularized<Loss>::dim;
  using PwVilmaRegularized<Loss>::loss_;
  using PwVilmaRegularized<Loss>::kPW;
  using PwVilmaRegularized<Loss>::alpha_buffer_;
  using PwVilmaRegularized<Loss>::data_;
  using PwVilmaRegularized<Loss>::wx_buffer_;
};
}

#include "pw_mord_regularized.hpp"

#endif /* defined(__sparse_sgd__pw_mord_no_beta_bmrm_oracle__) */
