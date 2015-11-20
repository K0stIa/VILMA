//
//  tail_parameters_oracle.hpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef tail_parameters_oracle_h
#define tail_parameters_oracle_h

#include <vector>
#include <memory>

#include "accpm_parameters_builder.h"

namespace VilmaOracle {
class OrdinalRegression;
}

namespace VilmaAccpmOracle {

class TailParametersOptimizationEngine {
 public:
  TailParametersOptimizationEngine() = delete;
  TailParametersOptimizationEngine(
      VilmaOracle::OrdinalRegression *,
      VilmaAccpmOracle::AccpmParametersBuilder *accpm_builder);

  std::vector<double> Optimize();

 protected:
 private:
  VilmaOracle::OrdinalRegression *caller;
  std::unique_ptr<VilmaAccpmOracle::AccpmParametersBuilder> accpm_builder;
};  // end TailParameterOracle

}  // namespace VilmaAccpmOracle
#endif /* tail_parameters_oracle_h */
