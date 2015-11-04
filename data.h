/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __vilma__data__
#define __vilma__data__

#include <stdio.h>

#include "sparse_matrix.h"
#include "dense_vector.h"

struct Data {
  Vilma::SparseMatrix<double>* x = nullptr;
  Vilma::DenseVector<int>* yl = nullptr;
  Vilma::DenseVector<int>* yr = nullptr;
  Vilma::DenseVector<int>* y = nullptr;
  Vilma::DenseVector<int>* z = nullptr;
  int nz, ny;
  ~Data() {
    if (x != nullptr) delete x;
    if (yl != nullptr) delete yl;
    if (yr != nullptr) delete yr;
    if (y != nullptr) delete y;
    if (z != nullptr) delete z;
  }
};

bool LoadData(const std::string& filepath, Data* data,
              const int max_num_examples, const int num_supervised_examples);

bool LoadTxtData(const std::string& features_path,
                 const std::string& labeling_path, const int dim,
                 const int n_classes, Data* data);

#endif /* defined(__vilma__data__) */
