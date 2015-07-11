//
//  bmrm_gender_racle.h
//  vilma
//
//  Created by Kostia on 3/3/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#ifndef __vilma__bmrm_gender_racle__
#define __vilma__bmrm_gender_racle__

#include "../bmrm/bmrm_solver.h"
#include "data.h"
#include "vector_interface.h"

#include <stdio.h>
#include <fstream>

namespace VilmaOracle {

template <class Oracle>
class BmrmOracleWrapper : public BMRM_Solver {
 public:
  typedef Vilma::DenseVector<double> DenseVecD;
  typedef Vilma::DenseVectorView<double> DenseVecDView;
  BmrmOracleWrapper(Oracle *oracle, bool normalize)
      : oracle_(oracle),
        BMRM_Solver(oracle->GetOracleParamsDim()),
        normalize_(normalize) {}

  ~BmrmOracleWrapper() {}

  virtual double risk(const double *w, double *subgrad) {
    const int dim = oracle_->GetOracleParamsDim();
    assert(dim == BMRM_Solver::dim);
    const int nexamples = oracle_->GetDataNumExamples();
    DenseVecD params(dim, const_cast<double *>(w));
    DenseVecD gradient(dim, subgrad);
    gradient.fill(0);
    double obj = 0;
    for (int i = 0; i < nexamples; ++i) {
      double val = oracle_->UpdateGradient(params, i, &gradient);
      obj += val;
    }
    if (normalize_) {
      gradient.mul(1. / nexamples);
      obj /= nexamples;
    }

#ifdef __MY_DEBUG__
    static int iter = 0;
    {
      const std::string file_path_prefix =
          "/Users/Shared/research/code/python/jmlr/data/"
          "2014-12-17-MorphIntervalAnnot/hist/iter_";

      std::string filepath = file_path_prefix + std::to_string(iter++) + ".bin";
      std::ifstream file(filepath, std::ios::in | std::ios::binary);
      if (file) {
        double s_obj = Vilma::VectorInterface::read<double>(&file);
        DenseVecD s_params(&file);
        DenseVecD s_grad(&file);
        file.close();
        // compare vectors
        double df_p = 0, df_g = 0;
        for (int i = 0; i < dim; ++i) {
          double t = s_params.data_[i] - params.data_[i];
          df_p += t * t;
          t = s_grad.data_[i] - gradient.data_[i];
          df_g += t * t;
        }
        df_p = std::sqrt(df_p);
        df_g = std::sqrt(df_g);
        std::cout << "obj diff: " << fabs(obj - s_obj) << " w-w: " << df_p
                  << " g-g: " << df_g << std::endl;
      }
    }
#endif
    return obj;
  }

  double EvaluateModel(Data *data, double *params) {
    return oracle_->EvaluateModel(data, params);
  }

 private:
  Oracle *oracle_;
  const bool normalize_;
};
}

#endif /* defined(__vilma__bmrm_gender_racle__) */
