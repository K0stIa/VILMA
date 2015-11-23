/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef vilma_h
#define vilma_h

#include "vilma_regularized.h"
#include "tail_parameters_oracle.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class VILma : public VilmaRegularized<Loss> {
 public:
  VILma() = delete;

  VILma(Data *data);
  virtual ~VILma() = default;

  using VilmaRegularized<Loss>::GetOracleParamsDim;
  using VilmaRegularized<Loss>::GetOracleData;
  using VilmaRegularized<Loss>::ProjectData;
  using VilmaRegularized<Loss>::SingleExampleBestLabelLookup;
  using VilmaRegularized<Loss>::UpdateSingleExampleGradient;

  virtual double risk(const double *weights, double *subgrad) override;

  std::vector<double> Train() override;

 protected:
  using VilmaRegularized<Loss>::dim;
  Loss loss_;
  // Oracle is never an owner of Data
  using VilmaRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using VilmaRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#include "vilma.hpp"

#endif /* vilma_h */
