//
//  vilma_regularized.hpp
//  VILMA
//
//  Created by Kostia on 11/21/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef vilma_regularized_h
#define vilma_regularized_h

#include "mord_regularized.h"

class Data;

namespace VilmaOracle {

typedef Vilma::DenseVector<double> DenseVecD;

template <class Loss>
class VilmaRegularized : public MOrdRegularized<Loss> {
 public:
  VilmaRegularized() = delete;

  VilmaRegularized(Data *data);
  virtual ~VilmaRegularized() = default;

  using MOrdRegularized<Loss>::GetOracleParamsDim;
  using MOrdRegularized<Loss>::GetOracleData;
  using MOrdRegularized<Loss>::ProjectData;
  using MOrdRegularized<Loss>::Train;
  using MOrdRegularized<Loss>::risk;
  using MOrdRegularized<Loss>::SingleExampleBestLabelLookup;

  virtual double UpdateSingleExampleGradient(
      const DenseVecD &beta, double *const wx, const int example_idx,
      double *w_gradient, double *free_params_gradient) override;

 protected:
  using MOrdRegularized<Loss>::dim;
  using MOrdRegularized<Loss>::loss_;
  // Oracle is never an owner of Data
  using MOrdRegularized<Loss>::data_;
  // buffer to store results <w,x> for all x
  using MOrdRegularized<Loss>::wx_buffer_;
};
}

#include "vilma_regularized.hpp"

#endif /* vilma_regularized_h */
