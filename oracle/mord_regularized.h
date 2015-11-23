/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef mord_regularized_h
#define mord_regularized_h

#include "vilma_regularized.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class MOrdRegularized : public VilmaRegularized<Loss> {
 public:
  MOrdRegularized() = delete;

  MOrdRegularized(Data *data);
  virtual ~MOrdRegularized() = default;

  using VilmaRegularized<Loss>::GetOracleParamsDim;
  using VilmaRegularized<Loss>::GetOracleData;
  using VilmaRegularized<Loss>::ProjectData;
  using VilmaRegularized<Loss>::Train;
  using VilmaRegularized<Loss>::risk;
  using VilmaRegularized<Loss>::SingleExampleBestLabelLookup;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using VilmaRegularized<Loss>::dim;
  using VilmaRegularized<Loss>::loss_;
  // Oracle is never an owner of Data
  using VilmaRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using VilmaRegularized<Loss>::wx_buffer_;
};
}

#include "mord_regularized.hpp"

#endif /* mord_regularized_h */
