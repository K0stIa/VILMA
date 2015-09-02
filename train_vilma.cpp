//
//  train_vilma.cpp
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
void RunExperiment(const string &features_path, const string &labeling_path,
                   const string &model_path, const double lambda, const int dim,
                   const int n_classes, const int bmrm_buffer_size) {
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
  oracle.set_lambda(lambda);
  // set up bmrm orracle buffer size
  oracle.set_BufSize(bmrm_buffer_size);

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_w = oracle.learn();

  const int num_gender = (int)opt_w.size() / data.x->kCols;

  // project data
  DenseVecD W((int)opt_w.size(), &opt_w[0]);
  vector<double> wx(data.x->kRows * num_gender);
  Oracle::ProjectData(W, &data, &wx[0]);

  // Learn beta
  DenseVecD beta(data.ny * num_gender);
  Oracle::TrainBeta(&beta, &data, &wx[0]);

  // print learning time
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Learning took "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start)
                   .count() << "us.\n";

  // glue W and beta to single vector
  vector<double> opt_params(W.dim_ + beta.dim_);

  for (int i = 0; i < W.dim_; ++i) opt_params[i] = W[i];

  for (int i = 0; i < beta.dim_; ++i) {
    opt_params[W.dim_ + i] = beta[i];
    assert(W.dim_ + i < (int)opt_params.size());
  }

  // save weights
  {
    std::ofstream file(model_path, std::ios::out | std::ios::binary);
    int size = (int)opt_params.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(&opt_params[0]),
               size * sizeof(opt_params[0]));
    file.close();
  }
  std::cout << "Model has been saved\n";
}

int main(int argc, const char *argv[]) {
  assert(argc == 8);

  const string features_path = argv[1];
  const string labeling_path = argv[2];
  const string model_path = argv[3];
  const int dim = atoi(argv[4]);
  const int n_classes = atoi(argv[5]);
  const double lambda = atof(argv[6]);
  const int bmrm_buffer_size = atoi(argv[7]);

  RunExperiment<BmrmOracle::SingleGenderNoBetaBmrmOracle<Loss>>(
      features_path, labeling_path, model_path, lambda, dim, n_classes,
      bmrm_buffer_size);

  return 0;
}
