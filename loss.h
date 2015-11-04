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

#include <stdio.h>

namespace Vilma {

class MAELoss {
 public:
  inline int operator()(const int y, const int t) const {
    return y > t ? y - t : t - y;
  }
};

class ZOLoss {
 public:
  inline int operator()(const int y, const int t) const {
    return y == t ? 0 : 1;
  }
};
  
}

#endif /* defined(__vilma__loss__) */
