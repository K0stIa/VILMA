/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#include "data.h"
#include "dense_vector.h"
#include "loss.h"

#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "evaluators.hpp"

using namespace std;

using std::vector;
using std::string;

typedef Vilma::DenseVector<double> DenseVecD;

#ifdef LOCAL_HOST
static const string kExperimentDir =
    "/Users/kostia/cmpgrid/datagrid/experiments/jml/";
#else
static const string kExperimentDir =
    "/mnt/datagrid/personal/antonkos/experiments/jml/";
#endif

tuple<vector<double>, vector<double>, vector<double>> ExtractResults(
    VilmaEvaluators::ModelEvaluator *model_evaluator, const string kDataset,
    const string kOracleName, const int kAge, const int supervised) {
  const string LPIP = "lpip";
  const string MORPH = "morph";

  const std::map<string, string> full_name = {
      std::make_pair(LPIP, "2015-02-04-LPIP"),
      std::make_pair(MORPH, "2014-12-17-MorphIntervalAnnot")};

  const std::map<string, std::vector<int>> split = {
      std::make_pair(LPIP, std::vector<int>{3300, 6600, 11000, 16000, 21000}),
      std::make_pair(MORPH,
                     std::vector<int>{3300, 6600, 10000, 13000, 23000, 33000})};

  const string kInputDir = kExperimentDir + full_name.at(kDataset);

  const vector<int> fractions = split.at(kDataset);
  vector<double> fraction_errors;
  vector<double> fraction_stds;
  vector<double> fraction_val_errors;

  char buff[256];

  for (int fraction : fractions) {
    double fraction_error = 0;
    double fraction_second_moment = 0;
    double fraction_val_error = 0;
    for (int perm_id = 1; perm_id <= 3; ++perm_id) {
      double best_val_error = 1e60;
      double best_tst_error = 1e60;

      for (double lambda : {0.1, 0.01}) {
        const string data_dir = kInputDir + "/" + kDataset + "/range" +
                                to_string(kAge) + "/perm-" + to_string(perm_id);
        const string trn_filepath = data_dir + "/" + kDataset + "-trn.bin";
        const string val_filepath = data_dir + "/" + kDataset + "-val.bin";
        const string tst_filepath = data_dir + "/" + kDataset + "-tst.bin";
        sprintf(buff, "%.4f", lambda);
        const string result_filename =
            kInputDir + "/" + kOracleName + "/year-" + to_string(kAge) +
            "/fraction-" + to_string(fraction) + "/" + to_string(perm_id) +
            "-" + string(buff);

        Data data;
        std::cout << "Loading train data from file: " << trn_filepath << endl;
        LoadData(trn_filepath, &data, fraction, supervised);
        std::cout << "Data loaded\n";

        std::ifstream file(result_filename + ".bin",
                           std::ios::in | std::ios::binary);
        if (!file) {
          std::cout << result_filename << "not found\n";
          continue;
        }

        DenseVecD opt_params(&file);
        file.close();

        double s = 0;
        for (int i = 0; i < opt_params.dim_; ++i) s += opt_params.data_[i];
        cout << "S: " << s << endl;

        // compute train error

        double trn_error = model_evaluator->Evaluate(&data, opt_params.data_);
        std::cout << "trn error: " << trn_error << std::endl;

        Data val_data;
        std::cout << "Loading validation data from file: " << val_filepath
                  << endl;

        LoadData(val_filepath, &val_data, int(1e9), int(1e9));
        std::cout << "Validation data loaded\n";

        // compute validation error
        double val_error =
            model_evaluator->Evaluate(&val_data, opt_params.data_);
        std::cout << "val error: " << val_error << std::endl;

        Data tst_data;
        std::cout << "Loading test data from file: " << tst_filepath << endl;

        LoadData(tst_filepath, &tst_data, int(1e9), int(1e9));
        std::cout << "Validation data loaded\n";

        // compute validation error
        double tst_error =
            model_evaluator->Evaluate(&tst_data, opt_params.data_);
        std::cout << "tst error: " << tst_error << std::endl;

        if (best_val_error > val_error) {
          best_val_error = val_error;
          best_tst_error = tst_error;
        }
      }  // end lambda loop
      fraction_error += best_tst_error;
      fraction_second_moment += best_tst_error * best_tst_error;
      fraction_val_error += best_val_error;
    }  // end perms
    fraction_errors.push_back(fraction_error / 3);
    fraction_stds.push_back(
        sqrt(fraction_second_moment / 3 - fraction_error * fraction_error / 9));
    fraction_val_errors.push_back(fraction_val_error / 3);
  }

  return std::make_tuple(fraction_errors, fraction_stds, fraction_val_errors);
}

template <class Loss>
void BuildTable2(const string dataset, const string classifier_id) {
  // supervised 3300
  vector<vector<double>> errors_mae_3300, stds_mae_3300;

  vector<int> age_set = {5, 10, 20};

  const string MAE_3300_SingleGenderNoBetaBmrmOracle =
      classifier_id + "-" + Loss::name() + "-" + to_string(3300);

  std::cout << "Oracle: " << MAE_3300_SingleGenderNoBetaBmrmOracle << std::endl;

  std::unique_ptr<VilmaEvaluators::MOrdModelEvaluator<Loss>> model_evaluator(
      new VilmaEvaluators::MOrdModelEvaluator<Loss>);

  for (int age : age_set) {
    auto res = ExtractResults(model_evaluator.get(), dataset,
                              MAE_3300_SingleGenderNoBetaBmrmOracle, age, 3300);
    errors_mae_3300.push_back(get<0>(res));
    stds_mae_3300.push_back(get<1>(res));
  }
  // supervised 6600
  vector<vector<double>> errors_mae_6600, stds_mae_6600;
  const string MAE_6600_SingleGenderNoBetaBmrmOracle =
      classifier_id + "-" + Loss::name() + "-" + to_string(6600);
  std::cout << "Oracle: " << MAE_6600_SingleGenderNoBetaBmrmOracle << std::endl;
  for (int age : age_set) {
    auto res = ExtractResults(model_evaluator.get(), dataset,
                              MAE_6600_SingleGenderNoBetaBmrmOracle, age, 6600);
    errors_mae_6600.push_back(get<0>(res));
    stds_mae_6600.push_back(get<1>(res));
  }

  // supervised baseline
  const string MAE_supervised_SingleGenderNoBetaBmrmOracle =
      classifier_id + "-" + Loss::name() + "-baseline";
  vector<vector<double>> errors_mae_baseline, stds_mae_baseline;
  std::cout << "Oracle: " << MAE_supervised_SingleGenderNoBetaBmrmOracle
            << std::endl;
  for (int age : {5}) {
    auto res =
        ExtractResults(model_evaluator.get(), dataset,
                       MAE_supervised_SingleGenderNoBetaBmrmOracle, age, 3300);
    errors_mae_baseline.push_back(get<0>(res));
    stds_mae_baseline.push_back(get<1>(res));
  }

  std::cout << "Results: " << std::endl;
  // baseline
  for (int k = 0; k < errors_mae_baseline.size(); ++k) {
    const auto &errors = errors_mae_baseline[k];
    const auto &stds = stds_mae_baseline[k];
    for (size_t i = 0; i < errors.size(); ++i) {
      std::cout << errors.at(i) << " +- " << stds.at(i) << " ";
    }
    std::cout << std::endl;
  }
  // 3300
  for (int k = 0; k < errors_mae_3300.size(); ++k) {
    const auto &errors = errors_mae_3300[k];
    const auto &stds = stds_mae_3300[k];
    cout << age_set[k] << " ";
    for (size_t i = 0; i < errors.size(); ++i) {
      std::cout << errors.at(i) << " +- " << stds.at(i) << " ";
    }
    std::cout << std::endl;
  }
  // 6600
  for (int k = 0; k < errors_mae_6600.size(); ++k) {
    const auto &errors = errors_mae_6600[k];
    const auto &stds = stds_mae_6600[k];
    cout << age_set[k] << " ";
    for (size_t i = 0; i < errors.size(); ++i) {
      std::cout << errors.at(i) << " +- " << stds.at(i) << " ";
    }
    std::cout << std::endl;
  }
}

template <class Loss>
void BuildBaselineTable(const string dataset, const string classifier_id) {
  // supervised baseline
  const string MAE_supervised_SingleGenderNoBetaBmrmOracle =
      classifier_id + "-" + Loss::name() + "-baseline";
  vector<vector<double>> errors_mae_baseline, stds_mae_baseline,
      val_errors_mae_baseline;
  std::cout << "Oracle: " << MAE_supervised_SingleGenderNoBetaBmrmOracle
            << std::endl;

  std::unique_ptr<VilmaEvaluators::MOrdModelEvaluator<Loss>> model_evaluator(
      new VilmaEvaluators::MOrdModelEvaluator<Loss>);

  for (int age : {5}) {
    auto res =
        ExtractResults(model_evaluator.get(), dataset,
                       MAE_supervised_SingleGenderNoBetaBmrmOracle, age, 3300);
    errors_mae_baseline.push_back(get<0>(res));
    stds_mae_baseline.push_back(get<1>(res));
    val_errors_mae_baseline.push_back(get<2>(res));
  }

  std::cout << "Results: " << std::endl;
  // baseline
  for (int k = 0; k < errors_mae_baseline.size(); ++k) {
    const auto &errors = errors_mae_baseline[k];
    const auto &stds = stds_mae_baseline[k];
    for (size_t i = 0; i < errors.size(); ++i) {
      std::cout << errors.at(i) << " +- " << stds.at(i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Val results: " << std::endl;
  for (int k = 0; k < val_errors_mae_baseline.size(); ++k) {
    const auto &errors = val_errors_mae_baseline[k];
    for (size_t i = 0; i < errors.size(); ++i) {
      std::cout << errors.at(i) << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char *argv[]) {
#ifdef LOCAL_HOST
  const string dataset = "morph";
  const string classifier_id = "SingleGenderNoBetaBmrmOracle";
#else
  const string dataset = argv[1];
  if (dataset != "morph" && dataset != "lpip") {
    cout << "Error! Bad dataset name! Must be either morph or lpip.";
    return 0;
  }

  const string classifier_id = argv[2];
  const string classifier_name =
      classifier_id.substr(0, classifier_id.find('-'));

  if (classifier_id != "Vilma" && classifier_id != "SvorImc" &&
      classifier_id != "PwMOrd" && classifier_name != "PwMOrdReg" &&
      classifier_name != "PwVilmaReg" && classifier_id != "SvorImcReg" &&
      classifier_id != "MOrdReg") {
    cout << "Error! Bad dataset name! Must be either morph or lpip.";
    return 0;
  }

#endif

  // BuildTable2<Vilma::MAELoss>(dataset, classifier_id);
  BuildBaselineTable<Vilma::MAELoss>(dataset, classifier_id);

  return 0;
}
