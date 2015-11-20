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

#include "accpm_tail_parameters_oracle.h"
#include "accpm_parameters_builder.h"

namespace VilmaOracle {
class OrdinalRegression;
}

namespace VilmaAccpmOracle {

class TailParametersOptimizationEngine {
 public:
  TailParametersOptimizationEngine() = delete;
  TailParametersOptimizationEngine(
      VilmaAccpmOracle::AccpmTailParametersOracle *accpm_oracle,
      VilmaAccpmOracle::AccpmParametersBuilder *accpm_builder);

  std::vector<double> Optimize();

  void ResetAccpmTailParametersOracle(
      VilmaAccpmOracle::AccpmTailParametersOracle *new_accpm_oracle);

  void ResetAccpmParametersBuilder(
      VilmaAccpmOracle::AccpmParametersBuilder *new_accpm_parameter_builder);

 protected:
 private:
  std::unique_ptr<VilmaAccpmOracle::AccpmTailParametersOracle> accpm_oracle_;
  std::unique_ptr<VilmaAccpmOracle::AccpmParametersBuilder> accpm_builder_;
};  // end TailParameterOracle

}  // namespace VilmaAccpmOracle
#endif /* tail_parameters_oracle_h */
