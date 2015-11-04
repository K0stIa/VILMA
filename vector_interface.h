/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

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
