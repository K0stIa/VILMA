//
//  sparse_vector.cpp
//  sparse_sgd
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "sparse_vector.h"
#include <assert.h>

namespace Vilma {
template <class T>
template <class V>
T SparseVector<T>::dot(const V &v) const {
  T s = 0;
  // how else can we garantee same dimensionality ?
  assert(dim_ == v.dim_);
  for (int k = 0; k < non_zero_; ++k) {
    s += vals_[k] * v.data_[index_[k]];
  }
  return s;
}
}  // namespace end
