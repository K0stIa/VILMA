//
//  data.cpp
//  vilma
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "data.h"

#include <iostream>

bool LoadData(const std::string& filepath, Data* data,
              const int max_num_examples, const int num_supervised_examples) {
  if (data == nullptr) return false;

  std::ifstream file(filepath, std::ios::in | std::ios::binary);
  if (!file) return false;

  int dim = Vilma::VectorInterface::read<int>(&file);
  int n = Vilma::VectorInterface::read<int>(&file);

  std::cout << "File has " << n << " examples with dim=" << dim << std::endl;
  std::cout << " Vilma use " << max_num_examples << " with "
            << num_supervised_examples << " supervised examples" << std::endl;

  data->x =
      new Vilma::SparseMatrix<double>(std::min(max_num_examples, n), dim);

  for (int i = 0; i < n; ++i) {
    int non_zero = Vilma::VectorInterface::read<int>(&file);
    int* index = nullptr;
    double* vals = nullptr;

    if (i < max_num_examples) {
      index = new int[non_zero];
      vals = new double[non_zero];
    }

    for (int j = 0; j < non_zero; ++j) {
      int t1 = Vilma::VectorInterface::read<int>(&file);
      double t2 = Vilma::VectorInterface::read<double>(&file);
      if (i < max_num_examples) {
        index[j] = t1;
        vals[j] = t2;
      }
    }

    if (i < max_num_examples) {
      // put example to the file
      Vilma::SparseVector<double>* row = data->x->GetRow(i);
      row->AssignNonZeros(index, vals, non_zero, dim);
    }
  }

  if (data->x->IsCorrupted()) {
    std::cout << "Data is Corrupted!\n";
    assert(false);
  }

  data->yl = new Vilma::DenseVector<int>(&file);
  data->yr = new Vilma::DenseVector<int>(&file);
  data->y = new Vilma::DenseVector<int>(&file);
  data->z = new Vilma::DenseVector<int>(&file);
  data->nz = data->z->max() + 1;
  data->ny = data->y->max() + 1;

  // make some examples supervised
  for (int i = 0; i < std::min(data->x->kRows, num_supervised_examples); ++i) {
    data->yl->data_[i] = data->y->data_[i];
    data->yr->data_[i] = data->y->data_[i];
  }

  file.close();
  return true;
}