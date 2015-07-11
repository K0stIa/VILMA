//
//  VectorInterface.h
//  sparse_sgd
//
//  Created by Kostia on 2/26/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#ifndef __vilma__VectorInterface__
#define __vilma__VectorInterface__

#include <stdio.h>
#include <fstream>

namespace Vilma {
  
class VectorInterface {
 public:
  VectorInterface(const VectorInterface&) = delete;
  VectorInterface(const VectorInterface&&) = delete;
  VectorInterface& operator=(VectorInterface&& other) = delete;
  VectorInterface& operator=(const VectorInterface& other) = delete;

  virtual ~VectorInterface() {}

  template <class T>
  static T read(std::ifstream* const file) {
    T val;
    file->read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
  }

  template <class T>
  static T* readArray(std::ifstream* const file, int* size) {
    *size = read<int>(file);
    T* vals = new T[*size];
    file->read(reinterpret_cast<char*>(vals), (*size) * sizeof(T));
    return vals;
  }

 protected:
  VectorInterface() {}
};
}

#endif /* defined(__vilma__VectorInterface__) */
