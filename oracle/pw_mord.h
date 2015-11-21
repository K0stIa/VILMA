/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef pw_mord_hpp
#define pw_mord_hpp

#include "pw_mord_regularized.h"
#include "tail_parameters_oracle.h"

namespace VilmaOracle {

template <class Loss>
class PwMOrd : public PwMOrdRegularized<Loss> {
 public:
  PwMOrd() = delete;

  PwMOrd(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwMOrd() = default;

  using PwMOrdRegularized<Loss>::GetOracleData;
  using PwMOrdRegularized<Loss>::Train;
  using PwMOrdRegularized<Loss>::SingleExampleBestLabelLookup;
  using PwMOrdRegularized<Loss>::ProjectData;
  using PwMOrdRegularized<Loss>::BuildAlphas;
  using PwMOrdRegularized<Loss>::GetOracleParamsDim;
  using PwMOrdRegularized<Loss>::kPW;

  virtual int GetFreeParamsDim() override;

  virtual double risk(const double *weights, double *subgrad) override;

  virtual std::vector<double> Train() override;

 protected:
  using PwMOrdRegularized<Loss>::UpdateSingleExampleGradient;

  // Oracle is never an owner of Data
  using PwMOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using PwMOrdRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#include "pw_mord.hpp"

#endif /* pw_mord_hpp */
