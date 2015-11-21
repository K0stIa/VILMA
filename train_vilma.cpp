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

#include "oracle/vilma.h"

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

typedef DenseVec<double> DenseVecD;

typedef Vilma::MAELoss Loss;

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
  vector<double> opt_w = oracle.Train();
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

  RunExperiment<VilmaOracle::VILma<Loss>>(features_path, labeling_path,
                                          model_path, lambda, dim, n_classes,
                                          bmrm_buffer_size);

  return 0;
}
