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

  if (nrhs < 2) {
    mexPrintf("Wrong number of input parameters!\n");
    return;
  }

  for (int i = 1; i < nrhs; ++i) {
    if (mxIsComplex(prhs[i])) {
      mexPrintf("VILMA works only with real inputs");
    }
  }

  if (strcmp("train", cmd) == 0) {
    mwSize m = mxGetM(prhs[1]);
    mwSize n = mxGetN(prhs[1]);
    mwSize nz = mxGetNzmax(prhs[1]);
    double *sr = mxGetPr(prhs[1]);
    // double *si = mxGetPi(prhs[1]);
    mwIndex *irs = mxGetIr(prhs[1]);
    mwIndex *jcs = mxGetJc(prhs[1]);
    for (mwIndex i = 0; i < nz; ++i) {
      mexPrintf("x: %d, y: %d, V %f\n", int(irs[i]), int(jcs[i]), sr[i]);
    }
    // prhs[2] -- labels
    // prhs[3] -- labels
    // prhs[4] -- n_classes
    // prhs[5] -- lambda
    
    const int n_classes = mxGetScalar(prhs[4]);
    mexPrintf("n_classes: %d \n", n_classes);

    if (!mxIsDouble(prhs[5])) {
      mexPrintf("lambda should be a real number");
    }
    const double lambda = mxGetScalar(prhs[5]);
    mexPrintf("lambda: %f\n", lambda);
  }
}
