/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef __vilma__loss__
#define __vilma__loss__

#include <string>

namespace Vilma {

class ScalarLossInterface {
 public:
  virtual inline int operator()(const int, const int) const = 0;
  static std::string name() { return "ScalarLossInterface"; }
};

class MAELoss : public ScalarLossInterface {
 public:
  inline int operator()(const int y, const int t) const override {
    return y > t ? y - t : t - y;
  }
  static std::string name() { return "MAELoss"; }
};

class ZOLoss : public ScalarLossInterface {
 public:
  inline int operator()(const int y, const int t) const override {
    return y == t ? 0 : 1;
  }
  static std::string name() { return "MAELoss"; }
};
}

#endif /* defined(__vilma__loss__) */
