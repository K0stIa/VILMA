//
//  generate_txt_file.cpp
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

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

template <class T>
using DenseVec = Vilma::DenseVector<T>;

// create txt file each row of wich is
// y_l y_r idx_1:val_1 idx_2:val_2 ... idx_n:val_n
//
void GenerateTxt(const string &input_dir, const string &output_filename) {
  Data data;
  std::cout << "Loading train data from file: " << (input_dir + "-trn.bin")
            << endl;
  if (LoadData(input_dir + "-trn.bin", &data, 33000, 0)) {
    std::cout << "Data loaded.\n";
  } else {
    std::cout << "Failed to load data!\n";
    return;
  }
  const int dim = data.x->kRows;
  const int ny = data.ny;
  const int n_examples = data.x->kRows;

  std::ofstream feat_file(output_filename + "_features.txt", std::ios::out);
  std::ofstream suplab_file(output_filename + "_supervised_labeling.txt",
                            std::ios::out);
  std::ofstream parlab_file(output_filename + "_partial_labeling.txt",
                            std::ios::out);

  for (int i = 0; i < n_examples; ++i) {
    const int yl = data.yl->data_[i];
    const int yr = data.yr->data_[i];
    const int y = data.y->data_[i];
    auto x = data.x->GetRow(i);

    for (int k = 0; k < x->non_zero_; ++k) {
      feat_file << x->index_[k] << ":" << x->vals_[k];
      if (k + 1 < x->non_zero_) {
        feat_file << " ";
      } else {
        feat_file << endl;
      }
    }

    suplab_file << y << " " << y << endl;
    parlab_file << yl << " " << yr << endl;
  }

  feat_file.close();
  suplab_file.close();
  parlab_file.close();

  cout << "dim: " << dim << " ny: " << ny << endl;
}

int main(int argc, const char *argv[]) {
  const string input_dir = argv[1];
  const string output_dir = argv[2];

  GenerateTxt(input_dir, output_dir);

  return 0;
}