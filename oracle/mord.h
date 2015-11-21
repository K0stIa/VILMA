/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef mord_h
#define mord_h

#include "mord_regularized.h"
#include "tail_parameters_oracle.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class MOrd : public MOrdRegularized<Loss> {
 public:
  MOrd() = delete;

  MOrd(Data *data);
  virtual ~MOrd() = default;

  using MOrdRegularized<Loss>::GetOracleParamsDim;
  using MOrdRegularized<Loss>::GetOracleData;
  using MOrdRegularized<Loss>::ProjectData;
  using MOrdRegularized<Loss>::SingleExampleBestLabelLookup;
  using MOrdRegularized<Loss>::UpdateSingleExampleGradient;

  virtual double risk(const double *weights, double *subgrad) override;

  std::vector<double> Train() override;

 protected:
  using MOrdRegularized<Loss>::dim;
  Loss loss_;
  // Oracle is never an owner of Data
  using MOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using MOrdRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#include "mord.hpp"

#endif /* mord_h */
