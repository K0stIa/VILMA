//
//  SparseMatrix.cpp
//  sparse_sgd
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "sparse_matrix.h"

namespace Vilma {

template <class T>
SparseMatrix<T>::SparseMatrix(const int rows, const int cols)
    : kRows(rows), kCols(cols) {
  vectors_ = new SparseVector<T>[rows];
}

template <class T>
SparseMatrix<T>::~SparseMatrix() {
  Clear();
}

template <class T>
SparseVector<T>* SparseMatrix<T>::GetRow(const int i) {
  if (i < 0 || i >= kRows) throw int();
  return vectors_ + i;
}

template <class T>
void SparseMatrix<T>::Clear() {
  if (vectors_ != nullptr) {
    delete[] vectors_;
    vectors_ = nullptr;
  }
}
}  // end of namespace