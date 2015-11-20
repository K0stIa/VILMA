//
//  accpm_parameters_builder.hpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef accpm_parameters_builder_hpp
#define accpm_parameters_builder_hpp

#include "Parameters.h"

namespace VilmaAccpmOracle {

class AccpmParametersBuilder {
 public:
  virtual Accpm::Parameters* Build() = 0;
};

class VilmaAccpmParametersBuilder : public AccpmParametersBuilder {
 public:
  VilmaAccpmParametersBuilder() = delete;
  VilmaAccpmParametersBuilder(const int dim);

  virtual Accpm::Parameters* Build() override;

 private:
  const int kDim;
};

}  // namespace VilmaAccpmOracle
#endif /* accpm_parameters_builder_hpp */
