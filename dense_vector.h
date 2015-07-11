//
//  dense_vector.h
//  vilma
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#ifndef __vilma__dense_vector__
#define __vilma__dense_vector__

#include <stdio.h>
#include <fstream>
#include "vector_interface.h"
#include "sparse_vector.h"

namespace Vilma {

template <class T>
class DenseVector : public VectorInterface {
 public:
  DenseVector(const int dim, T *data)
      : dim_(dim), data_(data), destroy_array_(false) {}

  DenseVector(const int dim) : dim_(dim) {
    data_ = new T[dim];
    std::fill(data_, data_ + dim, 0);
    destroy_array_ = true;
  }
  DenseVector(std::ifstream *const file) {
    data_ = readArray<T>(file, &dim_);
    destroy_array_ = true;
  }

  virtual ~DenseVector() {
    if (destroy_array_) Clear();
  }

  /**
   * index access
   */
  T &operator[](int i) {
#ifdef USE_ASSERT
    assert(0 <= i && i < dim_);
#endif
    return data_[i];
  }

  const T &operator[](int i) const {
#ifdef USE_ASSERT
    assert(0 <= i && i < dim_);
#endif
    return data_[i];
  }

  /** First check if it is non empty */
  const T max() const {
    T v = data_[0];
    for (int i = 1; i < dim_; ++i) v = std::max(v, data_[i]);
    return v;
  }

  /** First check if it is non empty */
  const T sum() const {
    T v = data_[0];
    for (int i = 1; i < dim_; ++i) v += data_[i];
    return v;
  }

  /** First check if it is non empty */
  const T norm2() const {
    T v = data_[0] * data_[0];
    for (int i = 1; i < dim_; ++i) v += data_[i] * data_[i];
    return v;
  }

  void add_sparse(const SparseVector<T> &, const T &);
  void add_dense(const DenseVector<T> &, const T &);

  template <class scalar>
  void mul(const scalar &);

  void fill(const T &v) { std::fill(data_, data_ + dim_, v); }

  void swap(DenseVector<T> &v) {
    swap(dim_, v.dim_);
    swap(data_, v.data_);
    swap(destroy_array_, v.destroy_array_);
  }

  int dim_;
  T *data_ = nullptr;

 protected:
  DenseVector() = delete;

 private:
  bool destroy_array_ = false;
  void Clear() {
    if (data_ != nullptr) delete[] data_;
    data_ = nullptr;
  }
};

template <class T>
class DenseVectorView : public DenseVector<T> {
 public:
  // TODO: remove it

  // do not use this class
  // using DenseVector<T>::DenseVector;
  using DenseVector<T>::max;
  using DenseVector<T>::fill;

  using DenseVector<T>::add_sparse;
  using DenseVector<T>::add_dense;
  using DenseVector<T>::data_;
  using DenseVector<T>::dim_;

  DenseVectorView(const int) = delete;
  DenseVectorView(std::ifstream *const) = delete;
  DenseVectorView(const DenseVector<T> &dense_vector, int begin, int end)
      : DenseVector<T>(end - begin, dense_vector.data_ + begin) {
    //    data_ = dense_vector.data_ + begin;
    //    dim_ = end - begin;
  }
  virtual ~DenseVectorView() {}
};
}  // namespace end

#include "dense_vector.hpp"

#endif /* defined(__vilma__dense_vector__) */
