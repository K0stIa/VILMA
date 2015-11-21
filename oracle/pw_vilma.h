//
//  pw_vilma.hpp
//  VILMA
//
//  Created by Kostia on 11/21/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef pw_vilma_h
#define pw_vilma_h

#include "pw_vilma_regularized.h"

namespace VilmaOracle {

template <class Loss>
class PwVilma : public PwVilmaRegularized<Loss> {
 public:
  PwVilma() = delete;

  PwVilma(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwVilma() = default;

  using PwVilmaRegularized<Loss>::GetOracleData;
  using PwVilmaRegularized<Loss>::Train;
  using PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup;
  using PwVilmaRegularized<Loss>::ProjectData;
  using PwVilmaRegularized<Loss>::BuildAlphas;
  using PwVilmaRegularized<Loss>::GetOracleParamsDim;
  using PwVilmaRegularized<Loss>::kPW;

  virtual int GetFreeParamsDim() override;

  virtual double risk(const double *weights, double *subgrad) override;

  virtual std::vector<double> Train() override;

 protected:
  using PwVilmaRegularized<Loss>::UpdateSingleExampleGradient;

  // Oracle is never an owner of Data
  using PwVilmaRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using PwVilmaRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#include "pw_vilma.hpp"

#endif /* pw_vilma_h */
