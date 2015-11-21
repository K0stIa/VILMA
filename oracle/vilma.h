/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef vilma_hpp
#define vilma_hpp

#include "vilma_regularized.h"
#include "tail_parameters_oracle.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class Vilma : public VilmaRegularized<Loss> {
 public:
  Vilma() = delete;

  Vilma(Data *data);
  virtual ~Vilma() = default;

  using VilmaRegularized<Loss>::GetOracleParamsDim;
  using VilmaRegularized<Loss>::GetOracleData;
  using VilmaRegularized<Loss>::ProjectData;
  using VilmaRegularized<Loss>::SingleExampleBestLabelLookup;
  using VilmaRegularized<Loss>::UpdateSingleExampleGradient;

  virtual double risk(const double *weights, double *subgrad) override;

  std::vector<double> Train();

 protected:
  using VilmaRegularized<Loss>::dim;
  using VilmaRegularized<Loss>::loss_;
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

#endif /* vilma_hpp */
