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

#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* time_t, struct tm, difftime, time, mktime */
#include <cstring>

#include "bmrm_solver.h"
#include "libqp.h"

using namespace shogun;

BMRM_Solver::BMRM_Solver(const int dim) {
  set_TolRel(0.01);
  set_TolAbs(0.0);
  set_BufSize(4000);
  set_lambda(1);
  set_cleanICP(true);
  set_cleanAfter(100);
  set_K(0.4);
  set_Tmax(100);
  set_cp_models(1);
  set_verbose(true);

  this->dim = dim;
}

BMRM_Solver::~BMRM_Solver() {}

void BMRM_Solver::set_TolRel(double tolrel) { _TolRel = tolrel; }

void BMRM_Solver::set_TolAbs(double tolabs) { _TolAbs = tolabs; }

void BMRM_Solver::set_BufSize(int sz) { _BufSize = sz; }

void BMRM_Solver::set_lambda(double lambda) { _lambda = lambda; }

void BMRM_Solver::set_cleanICP(bool _cleanICP) { cleanICP = _cleanICP; }
void BMRM_Solver::set_cleanAfter(int _cleanAfter) { cleanAfter = _cleanAfter; }
void BMRM_Solver::set_K(double k) { K = k; }
void BMRM_Solver::set_Tmax(int tmax) { T_max = tmax; }
void BMRM_Solver::set_cp_models(int _cp_models) { cp_models = _cp_models; }
void BMRM_Solver::set_verbose(bool _verbose) { verbose = _verbose; }

std::vector<double> BMRM_Solver::learn() {
  std::vector<double> w(dim, 0);
  report = svm_bmrm_solver(&w[0], _TolRel, _TolAbs, _lambda, _BufSize, cleanICP,
                           cleanAfter, K, T_max, verbose);

  return w;
}

#define LIBBMRM_PLUS_INF (-log(0.0))
#define LIBBMRM_CALLOC(x, y) calloc(x, y)
#define LIBBMRM_FREE(x) free(x)
#define LIBBMRM_MEMCPY(x, y, z) memcpy(x, y, z)
#define LIBBMRM_INDEX(ROW, COL, NUM_ROWS) ((COL) * (NUM_ROWS) + (ROW))
#define LIBBMRM_ABS(A) ((A) < 0 ? -(A) : (A))
#define LIBBMRM_MEMMOVE(x, y, z) memmove(x, y, z)

struct bmrm_ll {
  /** Pointer to previous CP entry */
  bmrm_ll *prev;
  /** Pointer to next CP entry */
  bmrm_ll *next;
  /** Pointer to the real CP data */
  double *address;
  /** Index of CP */
  int idx;
};

class SparseMatrix {
 public:
  SparseMatrix(const int buffsize, const size_t dim)
      : kDim(dim), rows(buffsize, NULL) {}

  double *GetRow(int idx) {
    if (rows[idx] == NULL) {
      rows[idx] = new double[kDim];
    }
    return rows[idx];
  }

  ~SparseMatrix() {
    for (double *ptr : rows) {
      if (ptr != NULL) {
        delete[] ptr;
      }
    }
  }

 private:
  const size_t kDim;
  std::vector<double *> rows;
};

/** Add cutting plane
 *
 * @param tail 		Pointer to the last CP entry
 * @param map		Pointer to map storing info about CP physical memory
 * @param A			CP physical memory
 * @param free_idx	Index to physical memory where the CP data will be
 *stored
 * @param cp_data	CP data
 * @param dim		Dimension of CP data
 */

void add_cutting_plane(bmrm_ll **tail, bool *map, SparseMatrix &A, int free_idx,
                       double *cp_data, int dim) {
  if (free_idx == -1) {
    printf("Can not add new cutting plane\n");
    assert(free_idx != -1);
  }
  assert(map[free_idx]);

  LIBBMRM_MEMCPY(A.GetRow(free_idx), cp_data, dim * sizeof(double));
  map[free_idx] = false;

  bmrm_ll *cp = (bmrm_ll *)LIBBMRM_CALLOC(1, sizeof(bmrm_ll));

  if (cp == NULL) {
    printf("Out of memory.\n");
    return;
  }

  cp->address = A.GetRow(free_idx);
  cp->prev = *tail;
  cp->next = NULL;
  cp->idx = free_idx;
  (*tail)->next = cp;
  *tail = cp;
}

/** Remove cutting plane at given index
 *
 * @param head	Pointer to the first CP entry
 * @param tail	Pointer to the last CP entry
 * @param map	Pointer to map storing info about CP physical memory
 * @param icp	Pointer to inactive CP that should be removed
 */

void remove_cutting_plane(bmrm_ll **head, bmrm_ll **tail, bool *map,
                          double *icp) {
  bmrm_ll *cp_list_ptr = *head;

  while (cp_list_ptr->address != icp) {
    cp_list_ptr = cp_list_ptr->next;
  }

  if (cp_list_ptr == *head) {
    *head = (*head)->next;
    cp_list_ptr->next->prev = NULL;
  } else if (cp_list_ptr == *tail) {
    *tail = (*tail)->prev;
    cp_list_ptr->prev->next = NULL;
  } else {
    cp_list_ptr->prev->next = cp_list_ptr->next;
    cp_list_ptr->next->prev = cp_list_ptr->prev;
  }

  map[cp_list_ptr->idx] = true;
  LIBBMRM_FREE(cp_list_ptr);
}

/** Get cutting plane
 *
 * @param ptr 	Pointer to some CP entry
 * @return Pointer to cutting plane at given entry
 */
inline double *get_cutting_plane(bmrm_ll *ptr) { return ptr->address; }

/** Get index of free slot for new cutting plane
 *
 * @param map	Pointer to map storing info about CP physical memory
 * @param size	Size of the CP buffer
 * @return Index of unoccupied memory field in CP physical memory
 */
inline int find_free_idx(bool *map, int size) {
  for (int i = 0; i < size; ++i)
    if (map[i]) return i;
  return -1;
}

static const int QPSolverMaxIter = 10000000;
static const double epsilon = 0.0000000001;

static double *H;
static int BufSize;

/*----------------------------------------------------------------------
 Returns pointer at i-th column of Hessian matrix.
 ----------------------------------------------------------------------*/
static const double *get_col(int i) { return (&H[BufSize * i]); }

bmrm_return_value_T BMRM_Solver::svm_bmrm_solver(double *W, double TolRel,
                                                 double TolAbs, double _lambda,
                                                 int _BufSize, bool cleanICP,
                                                 int cleanAfter, double K,
                                                 int Tmax, bool verbose) {
  bmrm_return_value_T bmrm;
  libqp_state_T qp_exitflag = {0, 0, 0, 0};
  double *b, *beta, *diag_H, *prevW;
  double R, *subgrad, QPSolverTolRel, C = 1.0, wdist = 0.0;
  double rsum, sq_norm_W, sq_norm_Wdiff = 0.0;
  int *I, *ICPcounter, *ACPs, cntICP = 0, cntACP = 0;
  int S = 1;
  int nDim = (int)this->dim;
  double **ICPs;
  SparseMatrix A(_BufSize, this->dim);

  // CTime ttime;
  // double tstart, tstop;
  time_t tstart, tstop;

  int nCP_new = 0;

  bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list = NULL;
  double *A_1 = NULL, *A_2 = NULL, *H_buff;
  bool *map = NULL;

  // tstart=ttime.cur_time_diff(false);
  time(&tstart);

  BufSize = _BufSize;
  QPSolverTolRel = 1e-9;

  H = NULL;
  b = NULL;
  beta = NULL;
  subgrad = NULL;
  diag_H = NULL;
  I = NULL;
  ICPcounter = NULL;
  ICPs = NULL;
  ACPs = NULL;
  H_buff = NULL;
  prevW = NULL;

  H = (double *)LIBBMRM_CALLOC((size_t)BufSize * BufSize, sizeof(double));

  if (H == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  // A = (double *)LIBBMRM_CALLOC((size_t)nDim * BufSize, sizeof(double));

  b = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  if (b == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  beta = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  if (beta == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  subgrad = (double *)LIBBMRM_CALLOC((size_t)nDim, sizeof(double));

  if (subgrad == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  diag_H = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  if (diag_H == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  I = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  if (I == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  ICPcounter = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  if (ICPcounter == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  ICPs = (double **)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double *));

  if (ICPs == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  ACPs = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  if (ACPs == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  map = (bool *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(bool));

  if (map == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  memset((bool *)map, true, (size_t)BufSize);

  cp_list = (bmrm_ll *)LIBBMRM_CALLOC((size_t)1, sizeof(bmrm_ll));

  if (cp_list == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  /* Temporary buffers for ICP removal */
  H_buff = (double *)LIBBMRM_CALLOC((size_t)BufSize * BufSize, sizeof(double));

  if (H_buff == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  prevW = (double *)LIBBMRM_CALLOC((size_t)nDim, sizeof(double));

  if (prevW == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

  bmrm.hist_Fp = std::vector<double>(BufSize);
  bmrm.hist_Fd = std::vector<double>(BufSize);
  bmrm.hist_wdist = std::vector<double>(BufSize);

  /* Iinitial solution */
  R = (double)this->risk(W, subgrad);

  bmrm.nCP = 0;
  bmrm.nIter = 0;
  bmrm.exitflag = 0;

  b[0] = -R;

  /* Cutting plane auxiliary double linked list */

  LIBBMRM_MEMCPY(A.GetRow(0), subgrad, nDim * sizeof(double));
  map[0] = false;
  cp_list->address = A.GetRow(0);
  cp_list->idx = 0;
  cp_list->prev = NULL;
  cp_list->next = NULL;
  CPList_head = cp_list;
  CPList_tail = cp_list;

  /* Compute initial value of Fp, Fd, assuming that W is zero vector */

  sq_norm_W = 0;
  bmrm.Fp = R + 0.5 * _lambda * sq_norm_W;
  bmrm.Fd = -LIBBMRM_PLUS_INF;

  // tstop=ttime.cur_time_diff(false);
  time(&tstop);

  /* Verbose output */

  if (verbose)
    printf("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf\n", bmrm.nIter,
           difftime(tstop, tstart), bmrm.Fp, bmrm.Fd, R);

  /* store Fp, Fd and wdist history */
  bmrm.hist_Fp[0] = bmrm.Fp;
  bmrm.hist_Fd[0] = bmrm.Fd;
  bmrm.hist_wdist[0] = 0.0;

  /* main loop */

  while (bmrm.exitflag == 0) {
    // tstart=ttime.cur_time_diff(false);
    time(&tstart);
    bmrm.nIter++;

    /* Update H */

    if (bmrm.nCP > 0) {
      A_2 = get_cutting_plane(CPList_tail);
      cp_ptr = CPList_head;

      for (int i = 0; i < bmrm.nCP; ++i) {
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;
        rsum = 0.0;

        for (int j = 0; j < nDim; ++j) {
          rsum += A_1[j] * A_2[j];
        }

        H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)] = rsum / _lambda;
      }

      for (int i = 0; i < bmrm.nCP; ++i) {
        H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)] =
            H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)];
      }
    }

    rsum = 0.0;
    A_2 = get_cutting_plane(CPList_tail);

    for (int i = 0; i < nDim; ++i) rsum += A_2[i] * A_2[i];

    H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)] = rsum / _lambda;

    diag_H[bmrm.nCP] = H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];
    I[bmrm.nCP] = 1;

    bmrm.nCP++;
    beta[bmrm.nCP] = 0.0;  // [beta; 0]

    /* call QP solver */
    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
                                    bmrm.nCP, QPSolverMaxIter, 0.0,
                                    QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);

    bmrm.qp_exitflag = qp_exitflag.exitflag;

    /* Update ICPcounter (add one to unused and reset used)
     * + compute number of active CPs */
    bmrm.nzA = 0;

    for (int aaa = 0; aaa < bmrm.nCP; ++aaa) {
      if (beta[aaa] > epsilon) {
        ++bmrm.nzA;
        ICPcounter[aaa] = 0;
      } else {
        ICPcounter[aaa] += 1;
      }
    }

    /* W update */
    for (int i = 0; i < nDim; ++i) {
      rsum = 0.0;
      cp_ptr = CPList_head;

      for (int j = 0; j < bmrm.nCP; ++j) {
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;
        rsum += A_1[i] * beta[j];
      }

      W[i] = -rsum / _lambda;
    }

    /* risk and subgradient computation */
    R = (double)this->risk(W, subgrad);
    b[bmrm.nCP] = -R;
    add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize),
                      subgrad, nDim);

    sq_norm_W = 0.0;
    sq_norm_Wdiff = 0.0;

    for (int j = 0; j < nDim; ++j) {
      b[bmrm.nCP] += subgrad[j] * W[j];
      sq_norm_W += W[j] * W[j];
      sq_norm_Wdiff += (W[j] - prevW[j]) * (W[j] - prevW[j]);
    }

    bmrm.Fp = R + 0.5 * _lambda * sq_norm_W;
    bmrm.Fd = -qp_exitflag.QP;
    wdist = ::sqrt(sq_norm_Wdiff);

    /* Stopping conditions */

    if (bmrm.Fp - bmrm.Fd <= TolRel * LIBBMRM_ABS(bmrm.Fp)) bmrm.exitflag = 1;

    if (bmrm.Fp - bmrm.Fd <= TolAbs) bmrm.exitflag = 2;

    if (bmrm.nCP >= BufSize) bmrm.exitflag = -1;

    // tstop=ttime.cur_time_diff(false);
    time(&tstop);

    /* Verbose output */

    if (verbose)
      printf(
          "%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, "
          "nCP=%d, nzA=%d, QPexitflag=%d\n",
          bmrm.nIter, difftime(tstop, tstart), bmrm.Fp, bmrm.Fd,
          bmrm.Fp - bmrm.Fd, (bmrm.Fp - bmrm.Fd) / bmrm.Fp, R, bmrm.nCP,
          bmrm.nzA, qp_exitflag.exitflag);

    /* Realocate if there more cutting planes then buffer*/
    if (bmrm.hist_Fp.size() >= bmrm.nIter) {
      bmrm.hist_Fp.resize(int(1.5 * bmrm.nIter + 1));
      bmrm.hist_Fd.resize(int(1.5 * bmrm.nIter + 1));
      bmrm.hist_wdist.resize(int(1.5 * bmrm.nIter + 1));
    }

    /* Keep Fp, Fd and w_dist history */
    bmrm.hist_Fp[bmrm.nIter] = bmrm.Fp;
    bmrm.hist_Fd[bmrm.nIter] = bmrm.Fd;
    bmrm.hist_wdist[bmrm.nIter] = wdist;

    /* Check size of Buffer */

    if (bmrm.nCP >= BufSize) {
      bmrm.exitflag = -2;
      if (verbose) printf("Buffer exceeded.\n");
    }

    /* keep W (for wdist history track) */
    LIBBMRM_MEMCPY(prevW, W, nDim * sizeof(double));

    /* Inactive Cutting Planes (ICP) removal */
    if (cleanICP) {
      /* find ICP */
      cntICP = 0;
      cntACP = 0;
      cp_ptr = CPList_head;
      int tmp_idx = 0;

      while (cp_ptr != CPList_tail) {
        if (ICPcounter[tmp_idx++] >= cleanAfter) {
          ICPs[cntICP++] = cp_ptr->address;
        } else {
          ACPs[cntACP++] = tmp_idx - 1;
        }

        cp_ptr = cp_ptr->next;
      }

      /* do ICP removal */
      if (cntICP > 0) {
        nCP_new = bmrm.nCP - cntICP;

        for (int i = 0; i < cntICP; ++i) {
          tmp_idx = 0;
          cp_ptr = CPList_head;

          while (cp_ptr->address != ICPs[i]) {
            cp_ptr = cp_ptr->next;
            tmp_idx++;
          }

          remove_cutting_plane(&CPList_head, &CPList_tail, map, ICPs[i]);

          LIBBMRM_MEMMOVE(b + tmp_idx, b + tmp_idx + 1,
                          (bmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(beta + tmp_idx, beta + tmp_idx + 1,
                          (bmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(diag_H + tmp_idx, diag_H + tmp_idx + 1,
                          (bmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(I + tmp_idx, I + tmp_idx + 1,
                          (bmrm.nCP - tmp_idx) * sizeof(int));
          LIBBMRM_MEMMOVE(ICPcounter + tmp_idx, ICPcounter + tmp_idx + 1,
                          (bmrm.nCP - tmp_idx) * sizeof(int));
        }

        /* H */
        for (int i = 0; i < nCP_new; ++i) {
          for (int j = 0; j < nCP_new; ++j) {
            H_buff[LIBBMRM_INDEX(i, j, BufSize)] =
                H[LIBBMRM_INDEX(ACPs[i], ACPs[j], BufSize)];
          }
        }

        for (int i = 0; i < nCP_new; ++i)
          for (int j = 0; j < nCP_new; ++j)
            H[LIBBMRM_INDEX(i, j, BufSize)] =
                H_buff[LIBBMRM_INDEX(i, j, BufSize)];

        bmrm.nCP = nCP_new;
      }
    }
  } /* end of main loop */

  bmrm.hist_Fp.resize(bmrm.nIter);
  bmrm.hist_Fd.resize(bmrm.nIter);
  bmrm.hist_wdist.resize(bmrm.nIter);

  cp_ptr = CPList_head;

  while (cp_ptr != NULL) {
    cp_ptr2 = cp_ptr;
    cp_ptr = cp_ptr->next;
    LIBBMRM_FREE(cp_ptr2);
    cp_ptr2 = NULL;
  }

  cp_list = NULL;

cleanup:

  LIBBMRM_FREE(H);
  LIBBMRM_FREE(b);
  LIBBMRM_FREE(beta);
  //  LIBBMRM_FREE(A);
  LIBBMRM_FREE(subgrad);
  LIBBMRM_FREE(diag_H);
  LIBBMRM_FREE(I);
  LIBBMRM_FREE(ICPcounter);
  LIBBMRM_FREE(ICPs);
  LIBBMRM_FREE(ACPs);
  LIBBMRM_FREE(H_buff);
  LIBBMRM_FREE(map);
  LIBBMRM_FREE(prevW);

  if (cp_list) LIBBMRM_FREE(cp_list);

  return (bmrm);
}