/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef accpm_constrained_theta_tail_parameters_oracle_h
#define accpm_constrained_theta_tail_parameters_oracle_h

#include "accpm_tail_parameters_oracle.h"

namespace VilmaAccpmOracle {

class AccpmConstrainedThetaTailParametersOracle
    : public AccpmTailParametersOracle {
 public:
  virtual ~AccpmConstrainedThetaTailParametersOracle() = default;
  AccpmConstrainedThetaTailParametersOracle(
      VilmaOracle::OrdinalRegression *ord_oracle, double *wx_buffer,
      const int dim, const int dw = 1);

  // TODO: reimplement it taking into account base class
  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info) override;

 protected:
  using AccpmTailParametersOracle::ord_oracle;
  using AccpmTailParametersOracle::kDim;
  using AccpmTailParametersOracle::accpm_grad_vector_;
  using AccpmTailParametersOracle::params;
  using AccpmTailParametersOracle::gradient;
  using AccpmTailParametersOracle::wx_buffer_;
  using AccpmTailParametersOracle::kDw;
};
}

#endif /* accpm_constrained_theta_tail_parameters_oracle_h */
