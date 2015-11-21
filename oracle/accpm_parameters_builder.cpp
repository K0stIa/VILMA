/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "accpm_parameters_builder.h"

using namespace VilmaAccpmOracle;

VilmaAccpmParametersBuilder::VilmaAccpmParametersBuilder(const int dim)
    : kDim(dim) {}

Accpm::Parameters* VilmaAccpmParametersBuilder::Build() {
  Accpm::Parameters* accpm_params = new Accpm::Parameters;
  // setup ACCPM
  accpm_params->setIntParameter("NumVariables", kDim);
  accpm_params->setIntParameter("NumSubProblems", 1);

  accpm_params->setIntParameter("MaxOuterIterations", 2000);
  accpm_params->setIntParameter("MaxInnerIterations", 500);

  accpm_params->setIntParameter("Proximal", 1);
  accpm_params->setRealParameter("Tolerance", 1e-3);
  accpm_params->setIntParameter("Verbosity", 0);
  accpm_params->setRealParameter("ObjectiveLB", 0.0);

  accpm_params->setIntParameter("ConvexityCheck", 1);
  accpm_params->setIntParameter("ConvexityFix", 1);

  accpm_params->setIntParameter("DynamicRho", 0);

  accpm_params->setIntParameter("Ball", 0);
  // param.setRealParameter("RadiusBall", 300);

  accpm_params->setRealParameter("WeightEpigraphCutInit", 1);
  accpm_params->setRealParameter("WeightEpigraphCutInc", 0);

  accpm_params->setVariableLB(vector<double>(kDim, -100));
  accpm_params->setVariableUB(vector<double>(kDim, 100));

  return accpm_params;
}