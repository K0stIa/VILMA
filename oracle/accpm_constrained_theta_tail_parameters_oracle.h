//
//  accpm_constrained_theta_tail_parameters_oracle.hpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

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
      const int dim);

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
};
}

#endif /* accpm_constrained_theta_tail_parameters_oracle_h */
