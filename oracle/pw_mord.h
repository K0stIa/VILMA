//
//  pw_mord.hpp
//  VILMA
//
//  Created by Kostia on 11/21/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef pw_mord_hpp
#define pw_mord_hpp

#include "pw_mord_regularized.h"
#include "tail_parameters_oracle.h"

namespace VilmaOracle {

template <class Loss>
class PwMOrd : public PwMOrdRegularized<Loss> {
 public:
  PwMOrd() = delete;

  PwMOrd(Data *data, const std::vector<int> &cut_labels);
  virtual ~PwMOrd() = default;

  using PwMOrdRegularized<Loss>::GetOracleData;
  using PwMOrdRegularized<Loss>::Train;
  using PwMOrdRegularized<Loss>::SingleExampleBestLabelLookup;
  using PwMOrdRegularized<Loss>::ProjectData;
  using PwMOrdRegularized<Loss>::BuildAlphas;
  using PwMOrdRegularized<Loss>::GetOracleParamsDim;
  using PwMOrdRegularized<Loss>::kPW;

  virtual int GetFreeParamsDim() override;

  virtual double risk(const double *weights, double *subgrad) override;

  virtual std::vector<double> Train() override;

 protected:
  using PwMOrdRegularized<Loss>::UpdateSingleExampleGradient;

  // Oracle is never an owner of Data
  using PwMOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using PwMOrdRegularized<Loss>::wx_buffer_;
  // beta oracle
  VilmaAccpmOracle::TailParametersOptimizationEngine free_parameters_oracle_;

 private:
  DenseVecD beta_;
};
}

#include "pw_mord.hpp"

#endif /* pw_mord_hpp */
