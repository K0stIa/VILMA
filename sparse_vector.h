/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __sparse_sgd__sparse_vector__
#define __sparse_sgd__sparse_vector__

#include <stdio.h>
#include <iostream>
#include "vector_interface.h"

namespace Vilma {

template <class T>
class SparseVector : public VectorInterface {
 public:
  SparseVector(const int dim) : dim_(dim), non_zero_(0) {}

  SparseVector() : SparseVector(0) {}

  SparseVector(const int dim, const int non_zero)
      : dim_(dim), non_zero_(non_zero) {
    index_ = new int[non_zero];
    vals_ = new T[non_zero];
  }

  virtual ~SparseVector() { Clear(); }

  /**
   * Takes ownership over index, vals arrays
   */
  void AssignNonZeros(int *index, T *vals, const int non_zero, const int dim) {
    Clear();
    index_ = index;
    vals_ = vals;
    non_zero_ = non_zero;
    dim_ = dim;
  }

  bool IsCorrupted() const {
    bool corrupted = false;
    for (int i = 0; i < non_zero_; ++i)
      if (index_[i] < 0 || index_[i] >= dim_) {
        std::cout << "i: " << i << " index_" << index_[i] << std::endl;
        corrupted = true;
      }
    return corrupted;
  }

  int GetDim() const { return dim_; }

  template <class V>
  T dot(const V &v) const;

  const int *index_ = nullptr;
  T *vals_ = nullptr;
  int dim_;
  int non_zero_;

 private:
  void Clear() {
    if (vals_ != nullptr) delete[] vals_;
    if (index_ != nullptr) delete[] index_;
    vals_ = nullptr;
    index_ = nullptr;
  }
};
}

#include "sparse_vector.hpp"

#endif /* defined(__sparse_sgd__sparse_vector__) */
