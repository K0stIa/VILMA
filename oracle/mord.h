//
//  mord.h
//  VILMA
//
//  Created by Kostia on 11/20/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef mord_h
#define mord_h

#include "mord_regularized.h"
#include "tail_parameters_oracle.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class MOrd : public MOrdRegularized<Loss> {
 public:
  MOrd() = delete;

  MOrd(Data *data);
  virtual ~MOrd() = default;

  using MOrdRegularized<Loss>::GetOracleParamsDim;
  using MOrdRegularized<Loss>::GetOracleData;
  using MOrdRegularized<Loss>::ProjectData;
  using MOrdRegularized<Loss>::SingleExampleBestLabelLookup;
  using MOrdRegularized<Loss>::UpdateSingleExampleGradient;

  virtual double risk(const double *weights, double *subgrad) override;

  std::vector<double> Train();

 protected:
  using MOrdRegularized<Loss>::dim;
  Loss loss_;
  // Oracle is never an owner of Data
  using MOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using MOrdRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#endif /* mord_h */
