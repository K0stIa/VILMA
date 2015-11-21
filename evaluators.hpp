//
//  evaluators.hpp
//  VILMA
//
//  Created by Kostia on 11/21/15.
//  Copyright © 2015 Kostia. All rights reserved.
//

#ifndef evaluators_hpp
#define evaluators_hpp

#include <vector>

#include "oracle/ordinal_regression.h"
#include "oracle/mord_regularized.h"
#include "oracle/pw_vilma_regularized.h"

namespace VilmaEvaluators {

typedef Vilma::DenseVector<double> DenseVecD;

class ModelEvaluator {
 public:
  ModelEvaluator() = default;
  virtual ~ModelEvaluator() = default;

  virtual double Evaluate(Data *data, double *params) const = 0;
};

template <class Loss>
class OrdModelEvaluator : public ModelEvaluator {
 public:
  OrdModelEvaluator() = default;
  virtual ~OrdModelEvaluator() = default;

  virtual double Evaluate(Data *data, double *params) const {
    const int num_examples = data->GetDataNumExamples();
    const int dim_x = data->GetDataDim();
    Loss loss;
    int error = 0;
    DenseVecD theta(data->GetDataNumClasses() - 1, params + dim_x);
    for (int ex = 0; ex < num_examples; ++ex) {
      const Vilma::SparseVector<double> &x = *data->x->GetRow(ex);
      const int y = data->y->data_[ex];
      const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
      int pred_y = VilmaOracle::OrdinalRegression::SingleExampleBestLabelLookup(
          wx, theta, data->GetDataNumClasses());
      error += loss(y, pred_y);
    }

    return 1. * error / num_examples;
  }
};

template <class Loss>
class MOrdModelEvaluator : public ModelEvaluator {
 public:
  MOrdModelEvaluator() = default;
  virtual ~MOrdModelEvaluator() = default;

  double Evaluate(Data *data, double *params) const override {
    const int num_examples = data->GetDataNumExamples();
    const int dim_x = data->GetDataDim();
    Loss loss;
    int error = 0;
    DenseVecD beta(data->GetDataNumClasses(), params + dim_x);
    for (int ex = 0; ex < num_examples; ++ex) {
      const Vilma::SparseVector<double> &x = *data->x->GetRow(ex);
      const int y = data->y->data_[ex];
      const double wx = x.dot<DenseVecD>(DenseVecD(dim_x, params));
      auto ret =
          VilmaOracle::MOrdRegularized<Loss>::SingleExampleBestLabelLookup(
              wx, beta, 0, data->GetDataNumClasses() - 1, -1, &loss);
      int pred_y = std::get<1>(ret);
      error += loss(y, pred_y);
    }

    return 1. * error / num_examples;
  }
};

template <class Loss>
class PwMOrdModelEvaluator : public ModelEvaluator {
 public:
  PwMOrdModelEvaluator() = delete;
  virtual ~PwMOrdModelEvaluator() = default;

  PwMOrdModelEvaluator(const std::vector<int> &cut_labels) {
    cut_labels_ = cut_labels;
  }

  double Evaluate(Data *data, double *params) const override {
    const int num_examples = data->GetDataNumExamples();
    const int dim_x = data->GetDataDim();
    const int ny = data->GetDataNumClasses();
    Loss loss;
    int error = 0;

    const int kPW = (int)cut_labels_.size();
    double *alpha_buffer =
        VilmaOracle::PwVilmaRegularized<Loss>::BuildAlphas(cut_labels_, ny);

    std::vector<double> wx(dim_x * kPW);
    DenseVecD weights(dim_x * kPW, params);

    VilmaOracle::PwVilmaRegularized<Loss>::ProjectData(weights, data, &wx[0],
                                                       kPW);

    const double *beta = params + dim_x * kPW;

    for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
      const int gt_y = data->y->operator[](example_idx);

      const auto subproblem_res =
          VilmaOracle::PwVilmaRegularized<Loss>::SingleExampleBestLabelLookup(
              &wx[0] + example_idx * kPW, alpha_buffer, beta, 0, ny - 1, -1,
              kPW, nullptr);

      const int &best_y = std::get<1>(subproblem_res);

      error += loss(gt_y, best_y);
    }

    return 1. * error / num_examples;
  }

 protected:
  std::vector<int> cut_labels_;
};

}  // namespace

#endif /* evaluators_hpp */
