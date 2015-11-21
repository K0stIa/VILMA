/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include <vector>

#include "ordinal_regression.h"
#include "QpGenerator.h"

#include "tail_parameters_oracle.h"

using namespace VilmaAccpmOracle;

TailParametersOptimizationEngine::TailParametersOptimizationEngine(
    VilmaAccpmOracle::AccpmTailParametersOracle *accpm_oracle,
    VilmaAccpmOracle::AccpmParametersBuilder *accpm_builder)
    : accpm_oracle_(accpm_oracle), accpm_builder_(accpm_builder) {}

std::vector<double> TailParametersOptimizationEngine::Optimize() {
  std::unique_ptr<Accpm::Parameters> params(accpm_builder_->Build());

  int n = params->getIntParameter("NumVariables");

  std::vector<double> start(n, 0);
  params->setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  // AccpmTailParametersOracle f1(caller, n);
  Accpm::Oracle oracle(accpm_oracle_.get());

  Accpm::QpGenerator qpGen;
  qpGen.init(params.get(), &oracle);
  while (!qpGen.run()) {
  }

  const Accpm::AccpmVector &x = *qpGen.getQueryPoint();
  std::vector<double> vals(x.size());

  for (int i = 0; i < x.size(); ++i) {
    vals[i] = x(i);
  }

  qpGen.terminate();

  return vals;
}

void TailParametersOptimizationEngine::ResetAccpmTailParametersOracle(
    VilmaAccpmOracle::AccpmTailParametersOracle *new_accpm_oracle) {
  accpm_oracle_.reset(new_accpm_oracle);
}

void TailParametersOptimizationEngine::ResetAccpmParametersBuilder(
    VilmaAccpmOracle::AccpmParametersBuilder *new_accpm_parameter_builder) {
  accpm_builder_.reset(new_accpm_parameter_builder);
}
