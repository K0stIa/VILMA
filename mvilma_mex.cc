#include <cstdio>
#include <cstring>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Get the command string
  char cmd[64];
  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd))) {
    mexErrMsgTxt(
        "First input should be a command string less than 64 characters long.");
  }

  if (!strcmp("train", cmd)) {
    
  }
}
