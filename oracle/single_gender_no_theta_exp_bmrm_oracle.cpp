//
//  single_gender_no_theta_exp_bmrm_oracle.cpp
//  vilma
//
//  Created by Kostia on 8/4/15.
//  Copyright (c) 2015 Kostia. All rights reserved.
//

#include "single_gender_no_theta_exp_bmrm_oracle.h"

#include "data.h"
#include "sparse_vector.h"
#include "loss.h"

#include "Parameters.h"
#include "QpGenerator.h"

#include <iostream>

using namespace BmrmOracle;

template <class Loss>
SingleGenderNoThetaExpBmrmOracle<Loss>::SingleGenderNoThetaExpBmrmOracle(
    Data *data)
    : data_(data), BMRM_Solver(data->x->kCols), theta_(data->ny - 1) {
  wx_buffer_ = new double[data_->x->kRows];
}

template <class Loss>
SingleGenderNoThetaExpBmrmOracle<Loss>::~SingleGenderNoThetaExpBmrmOracle() {
  if (wx_buffer_ != nullptr) delete[] wx_buffer_;
}

template <class Loss>
int SingleGenderNoThetaExpBmrmOracle<Loss>::GetDataDim() {
  return data_->x->kCols;
}

template <class Loss>
int SingleGenderNoThetaExpBmrmOracle<Loss>::GetOracleParamsDim() {
  return GetDataDim();
}

template <class Loss>
int SingleGenderNoThetaExpBmrmOracle<Loss>::GetDataNumExamples() {
  return data_->x->kRows;
}

template <class Loss>
int SingleGenderNoThetaExpBmrmOracle<Loss>::GetDataNumAgeClasses() {
  return data_->ny;
}

template <class Loss>
double SingleGenderNoThetaExpBmrmOracle<Loss>::risk(const double *weights,
                                                    double *subgrad) {
  const int nexamples = GetDataNumExamples();
  DenseVecD params(dim, const_cast<double *>(weights));
  DenseVecD gradient(dim, subgrad);

  ProjectData(params, data_, wx_buffer_);

  gradient.fill(0);

  // TODO: train theta
  TrainTheta(&theta_, data_, wx_buffer_);

  double obj = 0;
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    double val = UpdateSingleExampleGradient(
        params, theta_, wx_buffer_[example_idx], example_idx, &gradient);
    obj += val;
  }
  // normalize
  gradient.mul(1. / nexamples);
  obj /= nexamples;

  return obj;
}

template <class Loss>
double SingleGenderNoThetaExpBmrmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &params, const DenseVecD &theta, const double wx,
    const int example_idx, DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
#endif

  double left_subproblem = 0;
  double right_subproblem = 0;

  //  if (gt_yl < data_->ny - 1) {
  //    left_subproblem = fmax(0.0, 1.0 + wx - theta[gt_yl]);
  //  }
  //
  //  if (gt_yr > 0) {
  //    right_subproblem = fmax(0.0, 1.0 - wx + theta[gt_yr - 1]);
  //  }
  //
  //  double coef = 0;
  //  if (left_subproblem > 0) {
  //    ++coef;
  //  }
  //  if (right_subproblem > 0) {
  //    --coef;
  //  }

  if (gt_yl > 0) {
    left_subproblem = fmax(0.0, 1.0 - wx + theta[gt_yl - 1]);
  }

  if (gt_yr < data_->ny - 1) {
    right_subproblem = fmax(0.0, 1.0 + wx - theta[gt_yr]);
  }

  double coef = 0;
  if (left_subproblem > 0) {
    --coef;
  }
  if (right_subproblem > 0) {
    ++coef;
  }

  if (coef) {
    // get reference on curent example
    const Vilma::SparseVector<double> *x = data_->x->GetRow(example_idx);
    // TODO: make method for this
    for (int i = 0; i < x->non_zero_; ++i) {
      gradient->data_[x->index_[i]] += coef * x->vals_[i];
    }
  }

  return left_subproblem + right_subproblem;
}

template <class Loss>
void SingleGenderNoThetaExpBmrmOracle<Loss>::ProjectData(
    const DenseVecD &params, Data *data, double *wx_buffer) {
#ifdef USE_ASSERT
  assert(wx_buffer_ != nullptr);
#endif
  const int nexamples = data->x->kRows;
  // precompute wx[example_idx] = <x, w>
  for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
    const Vilma::SparseVector<double> *x = data->x->GetRow(example_idx);
    wx_buffer[example_idx] = x->dot<DenseVecD>(params);
  }
}

template <class Loss>
void SingleGenderNoThetaExpBmrmOracle<Loss>::TrainTheta(DenseVecD *theta,
                                                        Data *data,
                                                        double *wx_buffer) {
  const int var_dim = theta->dim_;

  Accpm::Parameters param;  //(paramFile);

  param.setIntParameter("NumVariables", var_dim);
  param.setIntParameter("NumSubProblems", 1);

  param.setIntParameter("MaxOuterIterations", 2000);
  param.setIntParameter("MaxInnerIterations", 500);

  param.setIntParameter("Proximal", 1);
  param.setRealParameter("Tolerance", 1e-3);
  param.setIntParameter("Verbosity", 0);
  param.setRealParameter("ObjectiveLB", 0.0);

  param.setIntParameter("ConvexityCheck", 1);
  param.setIntParameter("ConvexityFix", 1);

  param.setIntParameter("DynamicRho", 0);

  param.setIntParameter("Ball", 0);
  // param.setRealParameter("RadiusBall", 300);

  param.setRealParameter("WeightEpigraphCutInit", 1);
  param.setRealParameter("WeightEpigraphCutInc", 0);

  param.setVariableLB(vector<double>(var_dim, -100));
  param.setVariableUB(vector<double>(var_dim, 100));

  // param.output(std::cout);

  int n = param.getIntParameter("NumVariables");

  vector<double> start(n, 0);
  param.setStartingPoint(start);
  // vector<double> center(n, 0.5);
  // param.setCenterBall(center);

  SingleGenderAuxiliaryThetaAccpmOracle<Loss> f1(data, wx_buffer);
  Accpm::Oracle accpm_oracle(&f1);

  Accpm::QpGenerator qpGen;
  qpGen.init(&param, &accpm_oracle);
  while (!qpGen.run()) {
  }
  // qpGen.output(std::cout);

  const Accpm::AccpmVector &x = *qpGen.getQueryPoint();

#ifdef USE_ASSERT
  assert(x.size() == GetDataNumAgeClasses());
#endif

  for (int i = 0; i < x.size(); ++i) {
    theta->operator[](i) = x(i);
  }

  qpGen.terminate();
}

template <class Loss>
double SingleGenderNoThetaExpBmrmOracle<Loss>::EvaluateModel(Data *data,
                                                             double *params) {
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
        SingleGenderNoThetaExpBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
            wx, theta, kData.ny, y);
    error += loss_(y, pred_y);
  }

  return 1. * error / num_examples;
}

template <class Loss>
int SingleGenderNoThetaExpBmrmOracle<Loss>::SingleExampleBestAgeLabelLookup(
    const double wx, const DenseVecD &theta, const int ny, const int gt_y) {
  int best_y = 0;
  for (int y = 0; y < ny - 1; ++y) {
    if (wx >= theta[y]) {
      best_y = y + 1;
      break;
    }
  }
#ifdef USE_ASSERT
  assert(best_y != -1);
#endif
  return best_y;
}

template <class Loss>
SingleGenderAuxiliaryThetaAccpmOracle<
    Loss>::SingleGenderAuxiliaryThetaAccpmOracle(Data *data, double *wx)
    : OracleFunction(),
      loss_(),
      kNumAgeClasses(data->ny),
      accpm_grad_vector_(data->ny - 1),
      params_(data->ny - 1),
      gradient_(data->ny - 1),
      wx_buffer_(wx),
      data_(data) {}

template <class Loss>
SingleGenderAuxiliaryThetaAccpmOracle<
    Loss>::~SingleGenderAuxiliaryThetaAccpmOracle() {}

template <class Loss>
int SingleGenderAuxiliaryThetaAccpmOracle<Loss>::eval(
    const Accpm::AccpmVector &y, Accpm::AccpmVector &functionValue,
    Accpm::AccpmGenMatrix &subGradients, Accpm::AccpmGenMatrix *info) {
  // call native oracle function
  for (int i = 0; i < kNumAgeClasses - 1; ++i) {
    params_[i] = y(i);
  }

  // check for feasibility
  bool feasibly = true;
  for (int i = 0; i < kNumAgeClasses - 2; ++i) {
    if (params_[i] > params_[i + 1]) {
      feasibly = false;
    }
  }

  if (feasibly) {
    const int nexamples = data_->x->kRows;

    gradient_.fill(0);
    double obj = 0;

    for (int example_idx = 0; example_idx < nexamples; ++example_idx) {
      double val = UpdateSingleExampleGradient(params_, wx_buffer_[example_idx],
                                               example_idx, &gradient_);
      obj += val;
    }
    // normalize
    gradient_.mul(1. / nexamples);
    obj /= nexamples;

    functionValue = obj;

    for (int i = 0; i < kNumAgeClasses - 1; ++i) {
      accpm_grad_vector_(i) = gradient_[i];
    }

    if (info != nullptr) {
      *info = 1;
    }
  } else {
    // not feasibile point, so we do a feasibility cut
    // in OBOE if we have a linear constraint d^T * y <= d
    // d^T * (y - x) + d^T * x - d <= 0, where x is a query point.
    // Thus oracle has to return vector d and scalar value d^T * x - d

    // find wrongly ordered thresholds, e.g. theta_y > theta_{y+1}

    // https://github.com/lolow/oboe-fix/blob/master/doc/userguide/OBOE-UserGuide.pdf
    // section 3.1.2

    int idx = -1;
    for (int i = 0; i < kNumAgeClasses - 2; ++i) {
      if (params_[i] > params_[i + 1]) {
        idx = i;
        break;
      }
    }
    assert(idx != -1);
    for (int i = 0; i < kNumAgeClasses - 1; ++i) {
      accpm_grad_vector_(i) = 0;
    }
    accpm_grad_vector_(idx + 1) = -1;
    accpm_grad_vector_(idx) = 1;

    // assigning scalar
    functionValue = params_[idx] - params_[idx + 1];

    // tell oracle it is  feasibility cut
    if (info != nullptr) {
      *info = 0;
    }
  }

  memcpy(subGradients.addr(), accpm_grad_vector_.addr(),
         sizeof(double) * accpm_grad_vector_.size());

  return 0;
}

template <class Loss>
double SingleGenderAuxiliaryThetaAccpmOracle<Loss>::UpdateSingleExampleGradient(
    const DenseVecD &theta, const double wx, const int example_idx,
    DenseVecD *gradient) {
  // extract example labels
  const int gt_yl = data_->yl->data_[example_idx];
  const int gt_yr = data_->yr->data_[example_idx];

#ifdef USE_ASSERT
  assert(0 <= yl && yl < kData.ny);
  assert(0 <= yr && yr < kData.ny);
  assert(yl <= yr);
#endif

  double obj = 0;

  if (gt_yl > 0) {
    double val = 1.0 - wx + theta[gt_yl - 1];
    if (val > 0) {
      obj += val;
      gradient->data_[gt_yl - 1] += 1;
    }
  }

  if (gt_yr < data_->ny - 1) {
    double val = 1.0 + wx - theta[gt_yr];
    if (val > 0) {
      obj += val;
      gradient->data_[gt_yr] -= 1;
    }
  }

  return obj;
}

template class BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Vilma::MAELoss>;
template class BmrmOracle::SingleGenderAuxiliaryThetaAccpmOracle<
    Vilma::MAELoss>;
template class BmrmOracle::SingleGenderNoThetaExpBmrmOracle<Vilma::ZOLoss>;
template class BmrmOracle::SingleGenderAuxiliaryThetaAccpmOracle<Vilma::ZOLoss>;
