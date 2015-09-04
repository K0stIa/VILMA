//
//  test_vilma.cpp
//  vilma
//
//  Created by Kostia on 9/2/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "sparse_matrix.h"
#include "dense_vector.h"
#include "data.h"
#include "loss.h"
#include "model_evaluator.h"
#include "sparse_vector.h"

#include "oracle/sgd_tight_gender_supervised_oracle.h"
#include "oracle/bmrm_oracle_wrapper.h"
#include "oracle/sgd_age_oracle.h"
#include "oracle/single_gender_no_beta_bmrm_oracle.h"
#include "oracle/single_gender_no_theta_exp_bmrm_oracle.h"

#include "Oracle.h"
#include "QpGenerator.h"
#include "Parameters.h"
#include "AccpmBlasInterface.h"

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace Accpm;
using namespace std;

template <class Loss>
using AgeOracle = VilmaOracle::AgeOracle<Loss>;

template <class Oracle>
using BmrmOracleWrapper = VilmaOracle::BmrmOracleWrapper<Oracle>;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

typedef DenseVec<double> DenseVecD;

typedef Vilma::MAELoss Loss;

typedef BmrmOracleWrapper<VilmaOracle::BetaAuxiliaryOracle<Loss>> ACCPMOracle;

typedef BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Loss> EXPOracle;

template <class Oracle>
void EvaluateExperiment(const string &features_path,
                        const string &labeling_path, const string &model_path,
                        const int dim, const int n_classes) {
  Data data;
  std::cout << "Loading train data from txt features&labeling files: "
            << features_path << " " << labeling_path << endl;
  if (LoadTxtData(features_path, labeling_path, dim, n_classes, &data)) {
    std::cout << "Data loaded.\n";
  } else {
    std::cout << "Failed to load data!\n";
    return;
  }

  Oracle oracle(&data);

  std::ifstream file(model_path, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << model_path << "not found\n";
    return;
  }

  DenseVecD opt_params(&file);
  file.close();

  double error = oracle.EvaluateModel(&data, opt_params.data_);
  std::cout << error << endl;
}

int main(int argc, const char *argv[]) {
  //  assert(argc == 6);

  const string features_path = argv[1];
  const string labeling_path = argv[2];
  const string model_path = argv[3];
  const int dim = atoi(argv[4]);
  const int n_classes = atoi(argv[5]);

  EvaluateExperiment<BmrmOracle::SingleGenderNoBetaBmrmOracle<Loss>>(
      features_path, labeling_path, model_path, dim, n_classes);

  return 0;
}
