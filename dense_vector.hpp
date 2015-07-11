//
//  dense_vector.cpp
//  vilma
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "dense_vector.h"

#include <emmintrin.h>
//#include <smmintrin.h>
//#include <nmmintrin.h>
//#include <tmmintrin.h>

namespace Vilma {

template <class T>
void DenseVector<T>::add_dense(const DenseVector<T> &v, const T &d) {
  assert(v.dim_ == dim_);
  // TODO:
  // http://stackoverflow.com/questions/13071340/how-to-check-if-two-types-are-same-at-compiletimebonus-points-if-it-works-with
  // http://stackoverflow.com/questions/16893992/check-if-type-can-be-explicitly-converted
  // SSE sppedup??

  auto add_dummy = [](double *a, double *b, double d, int n) {
    for (int i = 0; i < n; ++i) {
      a[i] += d * b[i];
    }
  };

  auto add_sse2 = [](double *a, double *b, double d, int n) {
    int i = 0;
    double dd[2] = {d, d};
    __m128d dd2 = _mm_loadu_pd(dd);
    for (; i < n; i += 2) {
      __m128d a2 = _mm_loadu_pd(a + i);
      __m128d b2 = _mm_loadu_pd(b + i);
      __m128d sum = _mm_add_pd(a2, _mm_mul_pd(dd2, b2));
      _mm_storeu_pd(a + i, sum);
    }
    for (; i < n; i++) {
      a[i] += b[i] * d;
    }
  };

  add_dummy(data_, v.data_, d, dim_);
}

template <class T>
void DenseVector<T>::add_sparse(const SparseVector<T> &v, const T &d) {
  assert(v.dim_ == dim_);
  // TODO:
  // http://stackoverflow.com/questions/13071340/how-to-check-if-two-types-are-same-at-compiletimebonus-points-if-it-works-with
  // http://stackoverflow.com/questions/16893992/check-if-type-can-be-explicitly-converted
  // SSE sppedup??
  for (int i = 0; i < v.non_zero_; ++i) {
    assert(0 <= v.index_[0] && v.index_[i] < v.dim_);
    data_[v.index_[i]] += d * v.vals_[i];
  }
}
template <class T>
template <class scalar>
void DenseVector<T>::mul(const scalar &d) {
  // SSE

  auto mul_sse2 = [](double *a, double d, int n) {
    int i = 0;
    double dd[2] = {d, d};
    __m128d dd2 = _mm_loadu_pd(dd);
    for (; i < n; i += 2) {
      __m128d a2 = _mm_loadu_pd(a + i);
      __m128d m = _mm_mul_pd(dd2, a2);
      _mm_storeu_pd(a + i, m);
    }
    for (; i < n; i++) {
      a[i] *= d;
    }
  };

  auto mul_dummy = [](double *a, double d, int n) {
    for (int i = 0; i < n; ++i) {
      a[i] *= d;
    }
  };

  mul_dummy(data_, d, dim_);
}

}  // namespace end