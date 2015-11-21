/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>

#include "sparse_matrix.h"
#include "dense_vector.h"
#include "data.h"
#include "loss.h"
#include "model_evaluator.h"
#include "sparse_vector.h"
#include "oracle/vilma.h"

#include "evaluators.hpp"

using namespace std;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

typedef DenseVec<double> DenseVecD;

typedef Vilma::MAELoss Loss;

template <class Oracle>
void EvaluateExperiment(VilmaEvaluators::ModelEvaluator *model_evaluator,
                        const string &features_path,
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
  oracle.Train();

  std::ifstream file(model_path, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << model_path << "not found\n";
    return;
  }

  DenseVecD opt_params(&file);
  file.close();

  double error = model_evaluator->Evaluate(&data, opt_params.data_);
  std::cout << error << endl;
}

int main(int argc, const char *argv[]) {
  //  assert(argc == 6);

  const string features_path = argv[1];
  const string labeling_path = argv[2];
  const string model_path = argv[3];
  const int dim = atoi(argv[4]);
  const int n_classes = atoi(argv[5]);

  unique_ptr<VilmaEvaluators::ModelEvaluator> model_evaluator(
      new VilmaEvaluators::MOrdModelEvaluator<Loss>);

  EvaluateExperiment<VilmaOracle::VILma<Loss>>(model_evaluator.get(),
                                               features_path, labeling_path,
                                               model_path, dim, n_classes);

  return 0;
}
