//
//  SparseMatrix.h
//  vilma
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#ifndef __vilma__SparseMatrix__
#define __vilma__SparseMatrix__

#include <stdio.h>

#include "sparse_vector.h"

namespace Vilma {

template <class T>
class SparseMatrix {
 public:
  SparseMatrix(const int rows, const int cols);
  ~SparseMatrix();

  SparseVector<T>* GetRow(const int i);

  bool IsCorrupted() const {
    bool corrupted = false;
    for (int i = 0; i < kRows; ++i) {
      if (vectors_[i].IsCorrupted ()) {
        corrupted = true;
        std::cout << "bad example: " << i << std::endl;
      }
    }
    return corrupted;
  }

  const int kRows, kCols;
  SparseVector<T>* vectors_ = nullptr;

 private:
  void Clear();
};
}

#include "sparse_matrix.hpp"

#endif /* defined(__vilma__SparseMatrix__) */
