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