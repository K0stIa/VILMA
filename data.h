//
//  data.h
//  vilma
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

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

#endif /* defined(__vilma__data__) */
