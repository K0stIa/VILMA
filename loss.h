//
//  loss.h
//  vilma


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
