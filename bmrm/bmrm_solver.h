/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Written (W) 2015 Kostiantyn Antoniuk
 * Copyright (C) 2012 Michal Uricar
 * Copyright (C) 2015 Kostiantyn Antoniuk
 */

#ifndef _BMRM_SOLVER
#define _BMRM_SOLVER

#include <vector>

typedef struct {
  int nIter;       /* number of iterations  */
  int nCP;         /* number of cutting planes */
  int nzA;         /* number of active cutting planes */
  double Fp;       /* primal objective value  */
  double Fd;       /* reduced (dual) objective value */
  int qp_exitflag; /* exitflag from the last call of the inner QP solver  */
  int exitflag;    /* 1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
                      2 .. ocas.Q_P - ocas.Q_D <= TolAbs
                      -1 .. ocas.nCutPlanes >= BufSize
                      -2 .. not enough memory for the solver */

  /** Track of Fp values in individual iterations */
  std::vector<double> hist_Fp;

  /** Track of Fd values in individual iterations */
  std::vector<double> hist_Fd;

  /** Track of w_dist values in individual iterations */
  std::vector<double> hist_wdist;

} bmrm_return_value_T;

class BMRM_Solver {
  static const int BMRM_USUAL = 1;
  static const int BMRM_PROXIMAL = 2;

  double _TolRel;
  double _TolAbs;
  double _lambda;
  int _BufSize;
  bool cleanICP;
  int cleanAfter;
  double K;
  int T_max;
  int cp_models;
  bool verbose;
  bmrm_return_value_T report;

  bmrm_return_value_T svm_bmrm_solver(double *W, double TolRel, double TolAbs,
                                      double _lambda, int _BufSize,
                                      bool cleanICP, int cleanAfter, double K,
                                      int Tmax, bool verbose);

 protected:
  int dim;

 public:
  BMRM_Solver(const int dim);
  std::vector<double> learn();

  // this function needs to be redefined so that after oracle call
  // in subgrad variable there will be an subgradient at point w and
  // it should return function value
  virtual double risk(const double *w, double *subgrad) = 0;

  virtual ~BMRM_Solver();

  void set_TolRel(double);
  void set_TolAbs(double);
  void set_BufSize(int);
  void set_lambda(double);
  double get_lambda() { return _lambda; }
  void set_cleanICP(bool _cleanICP);
  void set_cleanAfter(int _cleanAfter);
  void set_K(double);
  void set_Tmax(int);
  void set_cp_models(int);
  void set_verbose(bool);

  bmrm_return_value_T get_report() { return report; }
};

#endif  // _BMRM_SOLVER
