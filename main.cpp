/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "sparse_matrix.h"
#include "dense_vector.h"
#include "data.h"
#include "loss.h"
#include "model_evaluator.h"
#include "sparse_vector.h"

#include "oracle/ordinal_regression.h"
#include "oracle/svor_imc_reg.h"
#include "oracle/svor_imc.h"

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

// template <class Loss>
// using Oracle = SgdOracle::TightGenderSupervisedOracle<Loss>;

// template <class Loss>
// using AgeOracle = VilmaOracle::AgeOracle<Loss>;

// template <class Oracle>
// using BmrmOracleWrapper = VilmaOracle::BmrmOracleWrapper<Oracle>;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

typedef DenseVec<double> DenseVecD;

typedef Vilma::MAELoss Loss;

template <class Oracle, class Loss>
void RunExperiment(const string &input_dir, const string &output_filename,
                   const double lambda, const int supervised,
                   const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  LoadData(input_dir + "-trn.bin", &data, fraction, supervised);
  std::cout << "Data loaded\n";

  Oracle atomic_oracle(&data);
  //  VilmaOracle::BmrmOracleWrapper<Oracle> oracle(&atomic_oracle, true);
  Oracle oracle(&data);
  oracle.set_lambda(lambda);

  if (lambda >= 1.0) {
    oracle.set_BufSize(300);
  } else if (lambda >= 0.1) {
    oracle.set_BufSize(700);
  } else {
    oracle.set_BufSize(1500);
  }

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_params = oracle.Train();

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
  double trn_error = VilmaOracle::EvaluateModel<Loss>(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double val_error =
      VilmaOracle::EvaluateModel<Loss>(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";

  // compute validation error
  double tst_error =
      VilmaOracle::EvaluateModel<Loss>(&tst_data, &opt_params[0]);
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

  if (oracle_name == "SvorImcReg") {
    // RunExperimentTheta(input_dir, output_dir, lambda, supervised, fraction);
    RunExperiment<VilmaOracle::SvorImcReg, Vilma::MAELoss>(
        input_dir, output_dir, lambda, supervised, fraction);
  } else if (oracle_name == "SingleGenderNoBetaBmrmOracle") {
    //
    //    RunExperiment<BmrmOracle::SingleGenderNoBetaBmrmOracle<Loss>>(
    //        input_dir, output_dir, lambda, supervised, fraction);
  } else if (oracle_name == "SvorImc") {
    RunExperiment<VilmaOracle::SvorImc, Vilma::MAELoss>(
        input_dir, output_dir, lambda, supervised, fraction);

  } else if (oracle_name == "SingleGenderAgeBmrmOracle") {
    //
    //    RunExperimentWithFullReg<VilmaOracle::AgeOracle<Loss>>(
    //        input_dir, output_dir, lambda, supervised, fraction);
    cout << "Oracle " << oracle_name << " is not supported!" << endl;
  } else {
    cout << "Oracle " << oracle_name << " is not supported!" << endl;
  }

  return 0;
}
