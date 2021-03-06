/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "data.h"
#include "dense_vector.h"
#include "loss.h"
#include "model_evaluator.h"
#include "sparse_matrix.h"
#include "sparse_vector.h"

#include "oracle/mord.h"
#include "oracle/ordinal_regression.h"
#include "oracle/pw_mord.h"
#include "oracle/pw_mord_regularized.h"
#include "oracle/pw_vilma.h"
#include "oracle/svor_exp.h"
#include "oracle/svor_imc.h"
#include "oracle/svor_imc_reg.h"
#include "oracle/vilma.h"
#include "oracle/vilma_regularized.h"

#include "evaluators.hpp"

using namespace std;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

typedef DenseVec<double> DenseVecD;

template <class Loss>
using PwRegression = VilmaOracle::PwVilmaRegularized<Loss>;

///////// Oracle Builders

struct OracleBuilderInterface {
  virtual VilmaOracle::OrdinalRegression *Build(Data *data) const = 0;
};

template <class Oracle>
struct OracleBuilder : public OracleBuilderInterface {
  Oracle *Build(Data *data) const override { return new Oracle(data); }
};

template <class Oracle>
struct PwOracleBuilder : public OracleBuilderInterface {
  PwOracleBuilder() = delete;
  PwOracleBuilder(const std::vector<int> &cut_labels)
      : cut_labels_(cut_labels) {}
  Oracle *Build(Data *data) const override {
    return new Oracle(data, cut_labels_);
  }
  const std::vector<int> cut_labels_;
};

///////// END Oracle Builders

void RunExperiment(OracleBuilderInterface *oracle_builder,
                   VilmaEvaluators::ModelEvaluator *model_evaluator,
                   const string &input_dir, const string &output_filename,
                   const double lambda, const int supervised,
                   const int fraction) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  LoadData(input_dir + "-trn.bin", &data, fraction, supervised);
  std::cout << "Data loaded\n";

  const int kNy = data.ny;
  assert(kNy == 55 || kNy == 80);

  VilmaOracle::OrdinalRegression *oracle = oracle_builder->Build(&data);

  oracle->set_lambda(lambda);

  if (lambda >= 1.0) {
    oracle->set_BufSize(300);
  } else if (lambda >= 0.1) {
    oracle->set_BufSize(700);
  } else {
    oracle->set_BufSize(1500);
  }

  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_params = oracle->Train();

  // print learning time
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Learning took "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start)
                   .count()
            << "us.\n";

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
  double trn_error = model_evaluator->Evaluate(&data, &opt_params[0]);
  std::cout << "trn error: " << trn_error << std::endl;

  Data val_data;
  std::cout << "Loading validation data from file: " << (input_dir + "-val.bin")
            << endl;

  LoadData(input_dir + "-val.bin", &val_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";
  val_data.ny = kNy;

  // compute validation error
  double val_error = model_evaluator->Evaluate(&val_data, &opt_params[0]);
  std::cout << "val error: " << val_error << std::endl;

  Data tst_data;
  std::cout << "Loading test data from file: " << (input_dir + "-tst.bin")
            << endl;

  LoadData(input_dir + "-tst.bin", &tst_data, int(1e9), int(1e9));
  std::cout << "Validation data loaded\n";
  tst_data.ny = kNy;

  // compute validation error
  double tst_error = model_evaluator->Evaluate(&tst_data, &opt_params[0]);
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
  assert(argc >= 7);

  const string input_dir = argv[1];
  const string output_dir = argv[2];
  string oracle_name = argv[3];
  const int supervised = atoi(argv[4]);
  const int fraction = atoi(argv[5]);
  const double lambda = atof(argv[6]);

  oracle_name = oracle_name.substr(0, oracle_name.find('-'));

  const string dataname =
      input_dir.substr(input_dir.find_last_of('/') + 1, input_dir.size());

  std::vector<int> cut_labels;

  if (argc >= 8) {
    const int n_pieces = atoi(argv[7]);

    if (dataname == "morph") {
      switch (n_pieces) {
        case 3:
          cut_labels = {0, 20, 40, 54};
          break;

        case 4:
          cut_labels = {0, 14, 26, 38, 54};
          break;

        case 5:
          cut_labels = {0, 11, 22, 33, 44, 54};
          break;

        default:
          std::cout << "n_pieces: " << n_pieces << " is not defined!"
                    << std::endl;
          return 0;
      }
    } else if (dataname == "lpip") {
      switch (n_pieces) {
        case 3:
          cut_labels = {0, 25, 54, 79};
          break;

        case 4:
          cut_labels = {0, 20, 40, 60, 79};
          break;

        case 5:
          cut_labels = {0, 16, 32, 48, 64, 79};
          break;

        default:
          std::cout << "n_pieces: " << n_pieces << " is not defined!"
                    << std::endl;
          return 0;
      }
    } else {
      std::cerr << "no " << dataname << " database!\n";
      return 0;
    }
  }

  std::unique_ptr<VilmaEvaluators::ModelEvaluator> model_evaluator;
  std::unique_ptr<OracleBuilderInterface> oracle_builder;

  if (oracle_name == "Vilma") {
    model_evaluator.reset(
        new VilmaEvaluators::MOrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::VILma<Vilma::MAELoss>>);

  } else if (oracle_name == "VilmaZO") {
    model_evaluator.reset(
        +new VilmaEvaluators::MOrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::VILma<Vilma::ZOLoss>>);

  } else if (oracle_name == "VilmaReg") {
    model_evaluator.reset(
        new VilmaEvaluators::MOrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(
        new OracleBuilder<VilmaOracle::VilmaRegularized<Vilma::MAELoss>>);

  } else if (oracle_name == "SvorImc") {
    model_evaluator.reset(
        new VilmaEvaluators::OrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::SvorImc>);

  } else if (oracle_name == "SvorExpReg") {
    cout << "Oracle " << oracle_name << " is not yet supported!" << endl;
    return 0;

  } else if (oracle_name == "SvorImcReg") {
    model_evaluator.reset(
        new VilmaEvaluators::OrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::SvorImcReg>);

  } else if (oracle_name == "SvorExp") {
    model_evaluator.reset(
        new VilmaEvaluators::OrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::SvorExp>);

  } else if (oracle_name == "MOrd") {
    model_evaluator.reset(
        new VilmaEvaluators::MOrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(new OracleBuilder<VilmaOracle::MOrd<Vilma::MAELoss>>);

  } else if (oracle_name == "MOrdReg") {
    model_evaluator.reset(
        new VilmaEvaluators::MOrdModelEvaluator<Vilma::MAELoss>);
    oracle_builder.reset(
        new OracleBuilder<VilmaOracle::MOrdRegularized<Vilma::MAELoss>>);

  } else if (oracle_name == "PwVilma") {
    model_evaluator.reset(
        new VilmaEvaluators::PwMOrdModelEvaluator<Vilma::MAELoss>(cut_labels));
    oracle_builder.reset(
        new PwOracleBuilder<VilmaOracle::PwVilma<Vilma::MAELoss>>(cut_labels));

  } else if (oracle_name == "PwVilmaReg") {
    model_evaluator.reset(
        new VilmaEvaluators::PwMOrdModelEvaluator<Vilma::MAELoss>(cut_labels));
    oracle_builder.reset(
        new PwOracleBuilder<VilmaOracle::PwVilmaRegularized<Vilma::MAELoss>>(
            cut_labels));

  } else if (oracle_name == "PwMOrd") {
    model_evaluator.reset(
        new VilmaEvaluators::PwMOrdModelEvaluator<Vilma::MAELoss>(cut_labels));
    oracle_builder.reset(
        new PwOracleBuilder<VilmaOracle::PwMOrd<Vilma::MAELoss>>(cut_labels));

  } else if (oracle_name == "PwMOrdReg") {
    model_evaluator.reset(
        new VilmaEvaluators::PwMOrdModelEvaluator<Vilma::MAELoss>(cut_labels));
    oracle_builder.reset(
        new PwOracleBuilder<VilmaOracle::PwMOrdRegularized<Vilma::MAELoss>>(
            cut_labels));

  } else {
    cout << "Oracle " << oracle_name << " is not yet supported!" << endl;
    return 0;
  }

  RunExperiment(oracle_builder.get(), model_evaluator.get(), input_dir,
                output_dir, lambda, supervised, fraction);

  return 0;
}
