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

#include <iostream>
#include <iterator>
#include <vector>
#include <sstream>
#include <algorithm>

int Data::GetDataDim() { return x->kCols; }

int Data::GetDataNumExamples() { return x->kRows; }

int Data::GetDataNumClasses() { return ny; }

bool LoadData(const std::string& filepath, Data* data,
              const int max_num_examples, const int num_supervised_examples) {
  if (data == nullptr) return false;

  std::ifstream file(filepath, std::ios::in | std::ios::binary);
  if (!file) return false;

  int dim = Vilma::VectorInterface::read<int>(&file);
  int n = Vilma::VectorInterface::read<int>(&file);

  std::cout << "File has " << n << " examples with dim=" << dim << std::endl;
  std::cout << " Vilma will use " << max_num_examples << " with "
            << num_supervised_examples << " supervised examples" << std::endl;

  data->x = new Vilma::SparseMatrix<double>(std::min(max_num_examples, n), dim);

  for (int i = 0; i < n; ++i) {
    int non_zero = Vilma::VectorInterface::read<int>(&file);
    int* index = nullptr;
    double* vals = nullptr;

    if (i < max_num_examples) {
      index = new int[non_zero];
      vals = new double[non_zero];
    }

    for (int j = 0; j < non_zero; ++j) {
      int t1 = Vilma::VectorInterface::read<int>(&file);
      double t2 = Vilma::VectorInterface::read<double>(&file);
      if (i < max_num_examples) {
        index[j] = t1;
        vals[j] = t2;
      }
    }

    if (i < max_num_examples) {
      // put example to the file
      Vilma::SparseVector<double>* row = data->x->GetRow(i);
      row->AssignNonZeros(index, vals, non_zero, dim);
    }
  }

  if (data->x->IsCorrupted()) {
    std::cout << "Data is Corrupted!\n";
    assert(false);
  }

  data->yl = new Vilma::DenseVector<int>(&file);
  data->yr = new Vilma::DenseVector<int>(&file);
  data->y = new Vilma::DenseVector<int>(&file);
  data->z = new Vilma::DenseVector<int>(&file);
  data->nz = data->z->max() + 1;
  data->ny = data->y->max() + 1;

  // make some examples supervised
  for (int i = 0; i < std::min(data->x->kRows, num_supervised_examples); ++i) {
    data->yl->data_[i] = data->y->data_[i];
    data->yr->data_[i] = data->y->data_[i];
  }

  file.close();
  return true;
}

bool LoadTxtData(const std::string& features_path,
                 const std::string& labeling_path, const int dim,
                 const int n_classes, Data* data) {
  if (data == nullptr) return false;

  std::ifstream feat_file(features_path, std::ios::in);
  std::ifstream lab_file(labeling_path, std::ios::in);
  if (!feat_file || !lab_file) return false;

  //  data->x = new Vilma::SparseMatrix<double>(std::min(max_num_examples, n),
  //  dim);
  std::vector<int*> indexes;
  std::vector<double*> values;
  std::vector<int> non_zero;
  std::vector<int> lower_labeling, upper_labeling;

  for (std::string feat_line; getline(feat_file, feat_line);) {
    // read labels
    int yl, yr;
    lab_file >> yl >> yr;
    lower_labeling.push_back(yl);
    upper_labeling.push_back(yr);
    // read features
    // replace secicolons by space
    for (char& c : feat_line) {
      if (c == ':') c = ' ';
    }
    // split line on tokens
    std::istringstream iss(feat_line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
              std::istream_iterator<std::string>(),
              std::back_inserter<std::vector<std::string> >(tokens));
    // store indexes & values
    std::vector<int> idxs;
    std::vector<double> vals;
    for (size_t i = 0; i < tokens.size(); i += 2) {
      int idx = std::stoi(tokens[i]);
      double val = std::stod(tokens[i + 1]);
      idxs.push_back(idx);
      vals.push_back(val);
    }
    non_zero.push_back((int)idxs.size());
    int* idx = new int[idxs.size()];
    double* val = new double[idxs.size()];
    for (size_t i = 0; i < idxs.size(); ++i) {
      idx[i] = idxs[i];
      val[i] = vals[i];
    }
    indexes.push_back(idx);
    values.push_back(val);
  }

  feat_file.close();
  lab_file.close();

  // store everythin to data
  data->x =
      new Vilma::SparseMatrix<double>(static_cast<int>(values.size()), dim);

  for (size_t i = 0; i < values.size(); ++i) {
    Vilma::SparseVector<double>* new_row = data->x->GetRow((int)i);
    new_row->AssignNonZeros(indexes[i], values[i], non_zero[i], dim);
  }
  data->yl =
      new Vilma::DenseVector<int>(static_cast<int>(lower_labeling.size()));
  for (int i = 0; i < (int)lower_labeling.size(); ++i) {
    data->yl->data_[i] = lower_labeling[i];
  }
  data->yr =
      new Vilma::DenseVector<int>(static_cast<int>(upper_labeling.size()));
  for (int i = 0; i < (int)upper_labeling.size(); ++i) {
    data->yr->data_[i] = upper_labeling[i];
  }
  data->y =
      new Vilma::DenseVector<int>(static_cast<int>(upper_labeling.size()));
  for (int i = 0; i < (int)upper_labeling.size(); ++i) {
    data->y->data_[i] = upper_labeling[i];
  }
  data->z = nullptr;
  data->ny =
      std::max(
          *std::max_element(lower_labeling.begin(), lower_labeling.end()),
          *std::max_element(upper_labeling.begin(), upper_labeling.end())) +
      1;
  data->nz = -1;

  return true;
}
