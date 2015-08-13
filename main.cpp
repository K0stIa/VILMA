//
//  main.cpp
//  sparse_sgd
//
//  Created by Kostia on 2/26/15.
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

// template <class Loss>
// using Oracle = SgdOracle::TightGenderSupervisedOracle<Loss>;

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
void RunExperiment(const string &input_dir, const string &output_filename,
                   const double lambda, const int supervised,
                   const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  if (LoadData(input_dir + "-trn.bin", &data, fraction, supervised)) {
    std::cout << "Data loaded.\n";
  } else {
    std::cout << "Failed to load data!\n";
    return;
  }

  Oracle oracle(&data);
  oracle.set_lambda(lambda);

  if (lambda >= 1.0) {
    oracle.set_BufSize(300);
  } else if (lambda >= 0.1) {
    oracle.set_BufSize(600);
  } else {
    oracle.set_BufSize(1000);
  }

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
    std::ofstream file(output_filename + ".bin",
                       std::ios::out | std::ios::binary);
    int size = (int)opt_params.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(&opt_params[0]),
               size * sizeof(opt_params[0]));
    file.close();
  }
  std::cout << "Classifier weights saved\n";

  // compute train error
  double trn_error = oracle.EvaluateModel(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double val_error = oracle.EvaluateModel(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double tst_error = oracle.EvaluateModel(&tst_data, &opt_params[0]);
  std::cout << "tst error: " << tst_error << std::endl;

  // save errors
  {
    std::ofstream fout;
    fout.open(output_filename + ".txt", std::ofstream::out);

    fout << trn_error << " " << val_error << " " << tst_error << std::endl;

    fout.close();
  }
}

template <>
void RunExperiment<EXPOracle>(const string &input_dir,
                              const string &output_filename,
                              const double lambda, const int supervised,
                              const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  if (LoadData(input_dir + "-trn.bin", &data, fraction, supervised)) {
    std::cout << "Data loaded.\n";
  } else {
    std::cout << "Failed to load data!\n";
    return;
  }

  std::cout << "TRaining Exp Oracle\n";

  EXPOracle oracle(&data);
  oracle.set_lambda(lambda);

  if (lambda >= 1.0) {
    oracle.set_BufSize(300);
  } else if (lambda >= 0.1) {
    oracle.set_BufSize(600);
  } else {
    oracle.set_BufSize(1000);
  }

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_w = oracle.learn();

  const int num_gender = (int)opt_w.size() / data.x->kCols;

  // project data
  DenseVecD W((int)opt_w.size(), &opt_w[0]);
  vector<double> wx(data.x->kRows * num_gender);
  EXPOracle::ProjectData(W, &data, &wx[0]);

  assert(num_gender == 1);
  // Learn beta
  DenseVecD theta((data.ny - 1) * num_gender);
  EXPOracle::TrainTheta(&theta, &data, &wx[0]);

  // print learning time
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Learning took "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start)
                   .count() << "us.\n";

  // glue W and beta to single vector
  vector<double> opt_params(W.dim_ + theta.dim_);

  for (int i = 0; i < W.dim_; ++i) opt_params[i] = W[i];

  for (int i = 0; i < theta.dim_; ++i) {
    opt_params[W.dim_ + i] = theta[i];
    assert(W.dim_ + i < (int)opt_params.size());
  }

  // save weights
  {
    std::ofstream file(output_filename + ".bin",
                       std::ios::out | std::ios::binary);
    int size = (int)opt_params.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(&opt_params[0]),
               size * sizeof(opt_params[0]));
    file.close();
  }
  std::cout << "Classifier weights saved\n";

  // compute train error
  double trn_error = oracle.EvaluateModel(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double val_error = oracle.EvaluateModel(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double tst_error = oracle.EvaluateModel(&tst_data, &opt_params[0]);
  std::cout << "tst error: " << tst_error << std::endl;

  // save errors
  {
    std::ofstream fout;
    fout.open(output_filename + ".txt", std::ofstream::out);

    fout << trn_error << " " << val_error << " " << tst_error << std::endl;

    fout.close();
  }
}

void RunExperimentTheta(const string &input_dir, const string &output_filename,
                        const double lambda, const int supervised,
                        const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  if (LoadData(input_dir + "-trn.bin", &data, fraction, supervised)) {
    std::cout << "Data loaded.\n";
  } else {
    std::cout << "Failed to load data!\n";
    return;
  }

  std::cout << "TRaining Exp Oracle\n";

  BmrmOracle::SingleGenderNoBetaBmrmOracle<Vilma::MAELoss> mae_oracle(&data);
  mae_oracle.set_lambda(lambda);

  if (lambda >= 1.0) {
    mae_oracle.set_BufSize(300);
  } else if (lambda >= 0.1) {
    mae_oracle.set_BufSize(600);
  } else {
    mae_oracle.set_BufSize(1000);
  }

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_w = mae_oracle.learn();

  const int num_gender = (int)opt_w.size() / data.x->kCols;

  // project data
  DenseVecD W((int)opt_w.size(), &opt_w[0]);
  vector<double> wx(data.x->kRows * num_gender);
  BmrmOracle::SingleGenderNoBetaBmrmOracle<Vilma::MAELoss>::ProjectData(
      W, &data, &wx[0]);

  assert(num_gender == 1);
  // Learn beta
  DenseVecD theta((data.ny - 1) * num_gender);
  BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Vilma::MAELoss>::TrainTheta(
      &theta, &data, &wx[0]);

  // print learning time
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Learning took "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start)
                   .count() << "us.\n";

  // glue W and beta to single vector
  vector<double> opt_params(W.dim_ + theta.dim_);

  for (int i = 0; i < W.dim_; ++i) opt_params[i] = W[i];

  for (int i = 0; i < theta.dim_; ++i) {
    opt_params[W.dim_ + i] = theta[i];
    assert(W.dim_ + i < (int)opt_params.size());
  }

  //  // save weights
  //  {
  //    std::ofstream file(output_filename + ".bin",
  //                       std::ios::out | std::ios::binary);
  //    int size = (int)opt_params.size();
  //    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  //    file.write(reinterpret_cast<const char *>(&opt_params[0]),
  //               size * sizeof(opt_params[0]));
  //    file.close();
  //  }
  std::cout << "Classifier weights saved\n";

  // compute train error
  double trn_error = mae_oracle.EvaluateModel(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Vilma::MAELoss> oracle(&data);

  // compute validation error
  double val_error = oracle.EvaluateModel(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double tst_error = oracle.EvaluateModel(&tst_data, &opt_params[0]);
  std::cout << "tst error: " << tst_error << std::endl;

  //  // save errors
  //  {
  //    std::ofstream fout;
  //    fout.open(output_filename + ".txt", std::ofstream::out);
  //
  //    fout << trn_error << " " << val_error << " " << tst_error << std::endl;
  //
  //    fout.close();
  //  }
}

template <class Oracle>
void RunExperimentWithFullReg(const string &input_dir,
                              const string &output_filename,
                              const double lambda, const int supervised,
                              const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  LoadData(input_dir + "-trn.bin", &data, fraction, supervised);
  std::cout << "Data loaded\n";

  Oracle atomic_oracle(&data);
  VilmaOracle::BmrmOracleWrapper<Oracle> oracle(&atomic_oracle, true);
  oracle.set_lambda(lambda);

  if (lambda >= 1.0) {
    oracle.set_BufSize(300);
  } else if (lambda >= 0.1) {
    oracle.set_BufSize(700);
  } else {
    oracle.set_BufSize(5000);
  }

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_params = oracle.learn();

  // print learning time
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Learning took "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start)
                   .count() << "us.\n";

  // save weights
  {
    std::ofstream file(output_filename + ".bin",
                       std::ios::out | std::ios::binary);
    int size = (int)opt_params.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(&opt_params[0]),
               size * sizeof(opt_params[0]));
    file.close();
  }
  std::cout << "Classifier weights saved\n";

  // compute train error
  double trn_error = oracle.EvaluateModel(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double val_error = oracle.EvaluateModel(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double tst_error = oracle.EvaluateModel(&tst_data, &opt_params[0]);
  std::cout << "tst error: " << tst_error << std::endl;

  // save errors
  {
    std::ofstream fout;
    fout.open(output_filename + ".txt", std::ofstream::out);

    fout << trn_error << " " << val_error << " " << tst_error << std::endl;

    fout.close();
  }
}

int main(int argc, const char *argv[]) {
  assert(argc == 7);

  const string input_dir = argv[1];
  const string output_dir = argv[2];
  string oracle_name = argv[3];
  const int supervised = atoi(argv[4]);
  const int fraction = atoi(argv[5]);
  const double lambda = atof(argv[6]);

  oracle_name = oracle_name.substr(0, oracle_name.find('-'));

  if (true) {
    RunExperimentTheta(input_dir, output_dir, lambda, supervised, fraction);
  } else if (oracle_name == "SingleGenderNoBetaBmrmOracle") {
    //
    RunExperiment<BmrmOracle::SingleGenderNoBetaBmrmOracle<Loss>>(
        input_dir, output_dir, lambda, supervised, fraction);
  } else if (oracle_name == "SingleGenderNoThetaExpBmrmOracle") {
    //
    RunExperiment<EXPOracle>(input_dir, output_dir, lambda, supervised,
                             fraction);

  } else if (oracle_name == "SingleGenderAgeBmrmOracle") {
    //
    RunExperimentWithFullReg<VilmaOracle::AgeOracle<Loss>>(
        input_dir, output_dir, lambda, supervised, fraction);
  } else {
    cout << "Oracle " << oracle_name << " is not supported!" << endl;
  }

  return 0;
}
