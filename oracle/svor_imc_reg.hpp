//
//  svor_imc_reg.hpp
//  VILMA
//
//  Created by Kostia on 11/19/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef svor_imc_reg_hpp
#define svor_imc_reg_hpp

#include "../bmrm/bmrm_solver.h"
#include "dense_vector.h"

#include "Oracle.h"

#include <stdio.h>
#include <memory>

class Data;

namespace BmrmOracle {

typedef Vilma::DenseVector<double> DenseVecD;
typedef Vilma::DenseVectorView<double> DenseVecDView;

template <class Loss>
class SvorImcReg : public BMRM_Solver {
 public:
  SvorImcReg() = delete;
  virtual ~SvorImcReg() = default;

  SvorImcReg(Data *data);

  int GetDataDim();
  int GetOracleParamsDim();
  int GetDataNumExamples();
  int GetDataNumAgeClasses();

  virtual double risk(const double *weights, double *subgrad);

  static int SingleExampleBestLabelLookup(const double wx,
                                          const DenseVecD &theta, const int ny);

  static void ProjectData(const DenseVecD &params, Data *data,
                          double *wx_buffer);
  static void TrainTheta(DenseVecD *theta, Data *data, double *wx_buffer);

  double EvaluateModel(Data *data, double *params);

 protected:
  using BMRM_Solver::dim;

  /**
   * Single example gradient computation
   */

  double UpdateSingleExampleGradient(const DenseVecD &theta, const double wx,
                                     const int example_idx,
                                     DenseVecD *gradient);

  Loss loss_;
  // Oracle is never an owner of Data
  Data *data_ = nullptr;
  // buffer to store results <w,x> for all x
  std::unique_ptr<double[]> wx_buffer_;
};

}  // namespace BmrmOracle

#endif /* svor_imc_reg_hpp */
