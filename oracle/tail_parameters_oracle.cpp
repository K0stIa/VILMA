//
//  tail_parameters_oracle.cpp
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#include "tail_parameters_oracle.h"

#include "ordinal_regression.h"
#include "QpGenerator.h"
#include "Oracle.h"

using namespace VilmaAccpmOracle;

TailParametersOptimizationEngine::TailParametersOptimizationEngine(
    VilmaOracle::OrdinalRegression *ord,
    VilmaAccpmOracle::AccpmParametersBuilder *accpm_builder)
    : caller(ord), accpm_builder(accpm_builder) {}

//////////
class AccpmTailParametersOracle : public Accpm::OracleFunction {
 public:
  AccpmTailParametersOracle(VilmaOracle::OrdinalRegression *ord_oracle,
                            const int dim)
      : ord_oracle(ord_oracle),
        kDim(dim),
        accpm_grad_vector_(dim),
        params(dim),
        gradient(dim) {}

  virtual ~AccpmTailParametersOracle() = default;

  virtual int eval(const Accpm::AccpmVector &y,
                   Accpm::AccpmVector &functionValue,
                   Accpm::AccpmGenMatrix &subGradients,
                   Accpm::AccpmGenMatrix *info) {
    // call native oracle function
    for (int i = 0; i < kDim; ++i) {
      params[i] = y(i);
    }
    gradient.fill(0);

    const int nexamples = ord_oracle->GetOracleData()->GetDataNumExamples();
    double *wx_buffer_ = ord_oracle->GetWxBuffer();
    double obj = 0;

    for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
      double val = ord_oracle->UpdateSingleExampleGradient(
          params, wx_buffer_[example_idx], example_idx, nullptr,
          gradient.data_);

      obj += val;
    }
    // normalize
    gradient.mul(1. / nexamples);
    obj /= nexamples;

    functionValue = obj;

    for (int i = 0; i < kDim; ++i) {
      accpm_grad_vector_(i) = gradient[i];
    }

    if (info != nullptr) {
      *info = 1;
    }

    memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
           sizeof(double) * accpm_grad_vector_.size());

    return 0;
  }

 protected:
  VilmaOracle::OrdinalRegression *ord_oracle;
  const int kDim;
  Accpm::AccpmVector accpm_grad_vector_;
  Vilma::DenseVector<double> params, gradient;
};
/////////

std::vector<double> TailParametersOptimizationEngine::Optimize() {
  std::unique_ptr<Accpm::Parameters> params(accpm_builder->Build());

  int n = params->getIntParameter("NumVariables");

  vector<double> start(n, 0);
  params->setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  AccpmTailParametersOracle f1(caller, n);
  Accpm::Oracle accpm_oracle(&f1);

  Accpm::QpGenerator qpGen;
  qpGen.init(params.get(), &accpm_oracle);
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