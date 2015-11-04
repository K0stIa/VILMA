/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

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
