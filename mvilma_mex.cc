#include <cstdio>
#include <cstring>
#include "mex.h"
#include "data.h"
#include "sparse_vector.h"
#include "sparse_matrix.h"


bool BuildVilmaData(mxArray *sparse_mat, Data *data) {
  if (data == nullptr) return false;

  mwSize m_ = mxGetM(sparse_mat);
  mwSize n_ = mxGetN(sparse_mat);
  mwSize nz = mxGetNzmax(sparse_mat);
  double *sr = mxGetPr(sparse_mat);
  mwIndex *irs = mxGetIr(sparse_mat);
  mwIndex *jcs = mxGetJc(sparse_mat);

  for (mwIndex i = 0; i < nz; ++i) {
    mexPrintf("x: %d, y: %d, V %f\n", int(irs[i]), int(jcs[i]), sr[i]);
  }

  const int dim = static_cast<int>(m_);
  const int n = static_cast<int>(n_);

  std::cout << "File has " << n << " examples with dim=" << dim << std::endl;
  std::cout << " Vilma will use " << max_num_examples << " with "
            << num_supervised_examples << " supervised examples" << std::endl;

  data->x = new Vilma::SparseMatrix<double>(n, dim);

  for (int i = 0; i < n; ++i) {
    mwIndex begin = jcs[i];
    mwIndex end = jcs[i + 1];
    int non_zero = end - begin;
    assert(non_zero > 0);
    int *index = new int[non_zero];
    double *vals = new double[non_zero];

    for (int j = 0; j < non_zero; ++j) {
      index[j] = static_cast<int>(irc[j + begin]);
      vals[j] = sr[j + begin];
    }

    // put example to the file
    Vilma::SparseVector<double> *row = data->x->GetRow(i);
    row->AssignNonZeros(index, vals, non_zero, dim);
  }

  if (data->x->IsCorrupted()) {
    std::cout << "Data is Corrupted!\n";
    assert(false);
  }

  data->yl = nullptr;
  data->yr = nullptr;
  data->y = nullptr;
  data->z = nullptr;

//      new Vilma::DenseVector<int>(static_cast<int>(upper_labeling.size()));
//  for (int i = 0; i < (int)upper_labeling.size(); ++i) {
//    data->y->data_[i] = upper_labeling[i];
//  }
  data->ny = -1;
  data->nz = -1;

  return true;
}

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
