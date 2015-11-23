#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "mex.h"

#include "data.h"
#include "sparse_vector.h"
#include "sparse_matrix.h"
#include "dense_vector.h"
//#include "oracle/pw_mord_no_beta_bmrm_oracle.h"
#include "oracle/svor_exp.h"
#include "oracle/mord.h"
#include "oracle/vilma.h"
#include "oracle/svor_imc.h"
#include "oracle/pw_mord.h"
#include "loss.h"

typedef Vilma::MAELoss Loss;

typedef VilmaOracle::PwMOrd<Vilma::MAELoss> PwOracle;
// typedef BmrmOracle::PwSingleGenderNoBetaBmrmOracle<Vilma::MAELoss> PwOracle;
// typedef BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Vilma::MAELoss> SvorImc;

// MOrd and VILma realisations are swaped!!!! FIX THIS!
typedef VilmaOracle::VILma<Vilma::MAELoss> VilmaMae;

typedef Vilma::DenseVector<double> DenseVecD;

bool BuildVilmaData(const mxArray *sparse_mat, const mxArray *lower_labels,
                    const mxArray *upper_labels, const mxArray *expected_labels,
                    Data *data) {
  if (data == nullptr) return false;

  mwSize m_ = mxGetM(sparse_mat);
  mwSize n_ = mxGetN(sparse_mat);
  mwSize nz = mxGetNzmax(sparse_mat);
  double *sr = mxGetPr(sparse_mat);
  mwIndex *irs = mxGetIr(sparse_mat);
  mwIndex *jcs = mxGetJc(sparse_mat);

  const int dim = static_cast<int>(m_);
  const int n = static_cast<int>(n_);

  data->x = new Vilma::SparseMatrix<double>(n, dim);

  for (int i = 0; i < n; ++i) {
    mwIndex begin = jcs[i];
    mwIndex end = jcs[i + 1];
    int non_zero = end - begin;
    assert(non_zero > 0);
    int *index = new int[non_zero];
    double *vals = new double[non_zero];

    for (int j = 0; j < non_zero; ++j) {
      index[j] = static_cast<int>(irs[j + begin]);
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

  const mxArray *input_labels[] = {lower_labels, upper_labels, expected_labels};
  Vilma::DenseVector<int> **data_labels[] = {&data->yl, &data->yr, &data->y};
  data->ny = -1;

  for (int k = 0; k < 3; ++k) {
    if (input_labels[k] != nullptr) {
      int *labels = static_cast<int *>(mxGetData(input_labels[k]));
      Vilma::DenseVector<int> *dl = new Vilma::DenseVector<int>(n);
      *(data_labels[k]) = dl;
      for (int t = 0; t < n; ++t) {
        // vilma label indexin starts from zero !
        dl->operator[](t) = labels[t] - 1;
        data->ny = std::max(data->ny, labels[t]);
      }
    } else {
      *(data_labels[k]) = nullptr;
    }
  }

  data->z = nullptr;
  data->nz = -1;

  return true;
}

template <class Oracle>
std::vector<double> TrainPwClassifier(Data *data, const double lambda,
                                      const int bmrm_buffer_size,
                                      const std::vector<int> &cut_labels) {
  Oracle oracle(data, cut_labels);
  oracle.set_lambda(lambda);
  oracle.set_BufSize(bmrm_buffer_size);

  return oracle.Train();
}

template <class Oracle>
std::vector<double> TrainClassifier(Data *data, const double lamda,
                                    const int bmrm_buffer_size) {
  Oracle oracle(data);
  oracle.set_lambda(lamda);
  oracle.set_BufSize(bmrm_buffer_size);

  return oracle.Train();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  //  freopen("output.txt", "w", stdout);
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

  if (strcmp("vilma-pw-mae", cmd) == 0) {
    if (!mxIsInt32(prhs[2]) || !mxIsInt32(prhs[5])) {
      mexErrMsgTxt(
          "Input labelings(or cut_labels) are not instances of not int32 "
          "class.");
      return;
    }

    Data data;
    std::cout << "loading data\n";
    BuildVilmaData(prhs[1], prhs[2], prhs[2], prhs[2], &data);
    // prhs[1] -- sparse feature matrix
    // prhs[2] -- labels
    // prhs[3] -- labels
    // prhs[4] -- n_classes
    // prhs[5] -- lambda

    const int n_classes = mxGetScalar(prhs[3]);

    if (!mxIsDouble(prhs[4])) {
      mexPrintf("lambda should be a real number");
    }
    const double lambda = mxGetScalar(prhs[4]);
    const int bmrm_buffer_size = 500;

    mwSize n_cut_labels = mxGetM(prhs[5]);
    int *_cut_labels = static_cast<int *>(mxGetData(prhs[5]));

    std::vector<int> cut_labels(_cut_labels, _cut_labels + n_cut_labels);

    // check cut_labels
    bool valid = true;
    for (int i = 0; i < (int)cut_labels.size() - 1; ++i) {
      if (cut_labels[i] > cut_labels[i + 1]) {
        valid = false;
      }
    }
    valid &= cut_labels.size() >= 2;
    valid &= cut_labels[0] == 1;
    valid &= cut_labels.back() == data.ny;

    if (!valid) {
      mexPrintf(
          "Your cut labels are invalid. cut_labels should be increasing subset "
          "of {1,...,Y} with 1 and Y included.");
    }

    std::vector<double> opt_w = TrainPwClassifier<PwOracle>(
        &data, lambda, bmrm_buffer_size, cut_labels);

    if (nlhs >= 1) {
      plhs[0] = mxCreateDoubleMatrix((int)opt_w.size(), 1, mxREAL);
      double *w = mxGetPr(plhs[0]);
      for (int i = 0; i < (int)opt_w.size(); ++i) {
        w[i] = opt_w[i];
      }
    }

  } else if (strcmp("svorimc", cmd) == 0 || strcmp("vilma-mae", cmd) == 0) {
    if (!mxIsInt32(prhs[2]) || !mxIsInt32(prhs[3])) {
      mexErrMsgTxt(
          "Input labelings(or cut_labels) are not instances of not int32 "
          "class.");
      return;
    }

    Data data;
    BuildVilmaData(prhs[1], prhs[2], prhs[3], nullptr, &data);
    // prhs[1] -- sparse feature matrix
    // prhs[2] -- labels
    // prhs[3] -- labels
    // prhs[4] -- n_classes
    // prhs[5] -- lambda

    const int n_classes = mxGetScalar(prhs[4]);
    mexPrintf("n_classes: %d \n", n_classes);
    data.ny = n_classes;

    if (!mxIsDouble(prhs[5])) {
      mexPrintf("terminating... lambda should be a real number");
      return;
    }
    const double lambda = mxGetScalar(prhs[5]);
    mexPrintf("lambda: %f\n", lambda);

    const int bmrm_buffer_size = 500;
    std::vector<double> opt_w;
    if (strcmp("svorimc", cmd) == 0) {
      opt_w = TrainClassifier<VilmaOracle::SvorImc>(&data, lambda,
                                                    bmrm_buffer_size);
    } else if (strcmp("vilma-mae", cmd) == 0) {
      opt_w = TrainClassifier<VilmaMae>(&data, lambda, bmrm_buffer_size);
    }
    if (nlhs >= 1) {
      plhs[0] = mxCreateDoubleMatrix((int)opt_w.size(), 1, mxREAL);
      double *w = mxGetPr(plhs[0]);
      for (int i = 0; i < (int)opt_w.size(); ++i) {
        w[i] = opt_w[i];
      }
    }
  }
}
