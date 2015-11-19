//
//  svor_imc_reg.cpp
//  VILMA
//
//  Created by Kostia on 11/19/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#include "svor_imc_reg.hpp"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

using namespace BmrmOracle;

template <class Loss>
SvorImcReg<Loss>::SvorImcReg(Data *data)
    : data_(data), BMRM_Solver(data->x->kCols + data->ny - 1) {
  wx_buffer_.reset(new double[data_->x->kRows]);
}

template <class Loss>
int SvorImcReg<Loss>::GetDataDim() {
  return data_->x->kCols;
}

template <class Loss>
int SvorImcReg<Loss>::GetOracleParamsDim() {
  return GetDataDim() + data_->ny - 1;
}

template <class Loss>
int SvorImcReg<Loss>::GetDataNumExamples() {
  return data_->x->kRows;
}

template <class Loss>
int SvorImcReg<Loss>::GetDataNumAgeClasses() {
  return data_->ny;
}

template <class Loss>
double SvorImcReg<Loss>::risk(const double *weights, double *subgrad) {
  const int nexamples = GetDataNumExamples();
  DenseVecD gradient(dim, subgrad);
  gradient.fill(0);

  DenseVecD w(GetDataDim(), const_cast<double *>(weights));
  DenseVecD theta(data_->ny - 1, const_cast<double *>(weights) + GetDataDim());
  ProjectData(w, data_, wx_buffer_.get());

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(theta, wx_buffer_[example_idx],
                                             example_idx, &gradient);
    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

template <class Loss>
double SvorImcReg<Loss>::UpdateSingleExampleGradient(const DenseVecD &theta,
                                                     const double wx,
                                                     const int example_idx,
                                                     DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

  double obj = 0;
  int total_sgn = 0;

  for (int y = 0; y < data_->ny - 1; ++y) {
    if (gt_yl <= y && y < gt_yr) {
      continue;
    }
    int sgn = 0;
    if (y < gt_yl) {
      sgn = 1;
    }
    if (gt_yr <= y) {
      sgn = -1;
    }
    double score = 1.0 - (wx - theta[y]) * sgn;
    if (score > 0) {
      obj += score;
      gradient->data_[GetDataDim() + y] += sgn;
      total_sgn += sgn;
    }
  }

  if (total_sgn != 0) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // update data cmponent of gradient
    for (int i = 0; i < x->non_zero_; ++i) {
      gradient->data_[x->index_[i]] -= total_sgn * x->vals_[i];
    }
  }

  return obj;
}

template <class Loss>
void SvorImcReg<Loss>::ProjectData(const DenseVecD &params, Data *data,
                                   double *wx_buffer) {
  const int nexamples = data->x->kRows;
  // precompute wx[example_idx] = <x, w>
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    const Vilma::SparseVector<double> *x = data->x->GetRow(example_idx);
    wx_buffer[example_idx] = x->dot<DenseVecD>(params);
  }
}

template <class Loss>
double SvorImcReg<Loss>::EvaluateModel(Data *data, double *params) {
  const Data &kData = *data;
  const int num_examples = data->x->kRows;
  const int dim_x = GetDataDim();
  int error = 0;
  DenseVecD theta(data->ny - 1, params + dim_x);
  for (int ex = 0; ex < num_examples; ++ex) {
    const Vilma::SparseVector<double> &x = *kData.x->GetRow(ex);
    const int y = kData.y->data_[ex];
    const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
    int pred_y =
        SvorImcReg<Loss>::SingleExampleBestLabelLookup(wx, theta, kData.ny);
    error += loss_(y, pred_y);
  }

  return 1. * error / num_examples;
}

template <class Loss>
int SvorImcReg<Loss>::SingleExampleBestLabelLookup(const double wx,
                                                   const DenseVecD &theta,
                                                   const int ny) {
  int best_y = 0;
  for (int y = 0; y < ny - 1; ++y) {
    if (wx >= theta[y]) {
      ++best_y;
    }
  }
  return best_y;
}

template class BmrmOracle::SvorImcReg<Vilma::MAELoss>;
template class BmrmOracle::SvorImcReg<Vilma::ZOLoss>;
