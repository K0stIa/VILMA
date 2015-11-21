/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef vilma_regularized_h
#define vilma_regularized_h

#include "mord_regularized.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class VilmaRegularized : public MOrdRegularized<Loss> {
 public:
  VilmaRegularized() = delete;

  VilmaRegularized(Data *data);
  virtual ~VilmaRegularized() = default;

  using MOrdRegularized<Loss>::GetOracleParamsDim;
  using MOrdRegularized<Loss>::GetOracleData;
  using MOrdRegularized<Loss>::ProjectData;
  using MOrdRegularized<Loss>::Train;
  using MOrdRegularized<Loss>::risk;
  using MOrdRegularized<Loss>::SingleExampleBestLabelLookup;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using MOrdRegularized<Loss>::dim;
  using MOrdRegularized<Loss>::loss_;
  // Oracle is never an owner of Data
  using MOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using MOrdRegularized<Loss>::wx_buffer_;
};
}

#include "vilma_regularized.hpp"

#endif /* vilma_regularized_h */
