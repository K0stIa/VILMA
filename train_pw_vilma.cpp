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

#include "sparse_matrix.h"
#include "dense_vector.h"
#include "data.h"
#include "loss.h"
#include "sparse_vector.h"

#include "oracle/pw_mord_no_beta_bmrm_oracle.h"
#include "oracle/pw_single_gender_no_beta_bmrm_oracle.h"

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

  vector<int> cut;
  //  for (int i = 0; i < data.ny; i += 20) {
  //    cut.push_back(i);
  //  }
  cut.push_back(0);
  cut.push_back(data.ny / 2);
  if (cut.size() && cut.back() != data.ny - 1) {
    cut.push_back(data.ny - 1);
  }

  Oracle oracle(&data, cut);
  oracle.set_lambda(lambda);
  // set up bmrm orracle buffer size
  oracle.set_BufSize(bmrm_buffer_size);
  //
  // measure learning time
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  // train W weights
  vector<double> opt_w = oracle.learn();

  cout << "found weights\n";
  for (auto x : opt_w) cout << x << " ";
  cout << endl;
  cout << "Fine!\n";

  //  const int num_gender = (int)opt_w.size() / data.x->kCols;

  // project data
  DenseVecD W((int)opt_w.size(), &opt_w[0]);
  vector<double> wx(data.x->kRows * oracle.kPW);
  Oracle::ProjectData(W, &data, &wx[0], oracle.kPW);

  //  wx = {1.13274,   -0.243122, -0.889616, 0.152217,  -1.00439,   0.852173,
  //        -0.916493, -1.52826,  2.44476,   -0.816766, 0.00936711, 0.807399,
  //        -2.77964,  1.67649,   1.10315,   -1.63353,  0.0187342,  1.6148,
  //        0.101976,  -0.782401, 0.680425,  -0.631107, 2.20676,    -1.57565,
  //        -0.705508, 1.89915,   -1.19364,  -3.11826,  0.652697,   2.46557,
  //        -8.50651,  3.58595,   4.92056,   -6.37988,  2.68946,    3.69042,
  //        -3.88822,  1.02634,   2.86188,   -5.1843,   1.36845,    3.81585,
  //        -6.48037,  1.71056,   4.76981,   -5.82318,  0.175087,   5.64809,
  //        -3.9749,   0.535787,  3.43912,   -3.69655,  -0.7214,    4.41795};

  cout << "projected:\n";
  for (double x : wx) cout << x << ", ";
  cout << endl;

  // Learn beta
  DenseVecD beta(data.ny);
  Oracle::TrainBeta(&beta, &data, &wx[0], oracle.alpha_buffer_, oracle.kPW);

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
  RunExperiment<BmrmOracle::PwSingleGenderNoBetaBmrmOracle<Loss>>(
      features_path, labeling_path, model_path, lambda, dim, n_classes,
      bmrm_buffer_size);

  return 0;
}
