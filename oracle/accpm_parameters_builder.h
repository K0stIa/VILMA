/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

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
