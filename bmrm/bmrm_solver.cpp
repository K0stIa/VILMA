#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h> /* time_t, struct tm, difftime, time, mktime */
#include <assert.h>

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

std::vector<double> BMRM_Solver::learn(int algorithm) {
  std::vector<double> w(dim, 0);

  if (algorithm == BMRM_Solver::BMRM_USUAL) {
    report = svm_bmrm_solver(&w[0], _TolRel, _TolAbs, _lambda, _BufSize,
                             cleanICP, cleanAfter, K, T_max, verbose);
  } else if (algorithm == BMRM_Solver::BMRM_PROXIMAL) {
    report = svm_ppbm_solver(&w[0], _TolRel, _TolAbs, _lambda, _BufSize,
                             cleanICP, cleanAfter, K, T_max, verbose);
  } else {
    printf("no algorithm %d\n", algorithm);
  }

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

void add_cutting_plane(bmrm_ll **tail, bool *map, double *A, int free_idx,
                       double *cp_data, int dim) {
  assert(map[free_idx]);

  LIBBMRM_MEMCPY(A + free_idx * dim, cp_data, dim * sizeof(double));
  map[free_idx] = false;

  bmrm_ll *cp = (bmrm_ll *)LIBBMRM_CALLOC(1, sizeof(bmrm_ll));

  if (cp == NULL) {
    printf("Out of memory.\n");
    return;
  }

  cp->address = A + (free_idx * dim);
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
  return size + 1;
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
  double R, *subgrad, *A, QPSolverTolRel, C = 1.0, wdist = 0.0;
  double rsum, sq_norm_W, sq_norm_Wdiff = 0.0;
  int *I, *ICPcounter, *ACPs, cntICP = 0, cntACP = 0;
  int S = 1;
  int nDim = (int)this->dim;
  double **ICPs;

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
  A = NULL;
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

  A = (double *)LIBBMRM_CALLOC((size_t)nDim * BufSize, sizeof(double));

  if (A == NULL) {
    bmrm.exitflag = -2;
    goto cleanup;
  }

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

  LIBBMRM_MEMCPY(A, subgrad, nDim * sizeof(double));
  map[0] = false;
  cp_list->address = &A[0];
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
  LIBBMRM_FREE(A);
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

// proximal BMRM

static double *H2;

static const double *get_col_2(int i) { return (&H2[BufSize * i]); }

bmrm_return_value_T BMRM_Solver::svm_ppbm_solver(double *W, double TolRel,
                                                 double TolAbs, double _lambda,
                                                 int _BufSize, bool cleanICP,
                                                 int cleanAfter, double K,
                                                 int Tmax, bool verbose) {
  bmrm_return_value_T ppbmrm;
  libqp_state_T qp_exitflag = {0, 0, 0, 0}, qp_exitflag_good = {0, 0, 0, 0};
  double *b, *b2, *beta, *beta_good, *beta_start, *diag_H, *diag_H2;
  double R, *subgrad, *A, QPSolverTolRel, C = 1.0;
  double *prevW, *wt, alpha, alpha_start, alpha_good = 0.0, Fd_alpha0 = 0.0;
  double lastFp, wdist, gamma = 0.0;
  double rsum, sq_norm_W, sq_norm_Wdiff, sq_norm_prevW, eps;
  int *I, *I2, *I_start, *I_good, *ICPcounter, *ACPs, cntICP = 0, cntACP = 0;
  int S = 1;
  int nDim = (int)this->dim;
  double **ICPs;
  int nCP_new = 0, qp_cnt = 0;
  bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list = NULL;
  double *A_1 = NULL, *A_2 = NULL, *H_buff;
  bool *map = NULL, tuneAlpha = true, flag = true, alphaChanged = false,
       isThereGoodSolution = false;

  time_t tstart, tstop;

  time(&tstart);

  BufSize = _BufSize;
  QPSolverTolRel = 1e-9;

  H = NULL;
  b = NULL;
  beta = NULL;
  A = NULL;
  subgrad = NULL;
  diag_H = NULL;
  I = NULL;
  ICPcounter = NULL;
  ICPs = NULL;
  ACPs = NULL;
  prevW = NULL;
  wt = NULL;
  H_buff = NULL;
  diag_H2 = NULL;
  b2 = NULL;
  I2 = NULL;
  H2 = NULL;
  I_good = NULL;
  I_start = NULL;
  beta_start = NULL;
  beta_good = NULL;

  alpha = 0.0;

  H = (double *)LIBBMRM_CALLOC((size_t)BufSize * BufSize, sizeof(double));

  A = (double *)LIBBMRM_CALLOC((size_t)nDim * BufSize, sizeof(double));

  b = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  beta = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  subgrad = (double *)LIBBMRM_CALLOC((size_t)nDim, sizeof(double));

  diag_H = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  I = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  ICPcounter = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  ICPs = (double **)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double *));

  ACPs = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));
  cp_list = (bmrm_ll *)LIBBMRM_CALLOC((size_t)1, sizeof(bmrm_ll));

  prevW = (double *)LIBBMRM_CALLOC((size_t)nDim, sizeof(double));

  wt = (double *)LIBBMRM_CALLOC((size_t)nDim, sizeof(double));

  if (H == NULL || A == NULL || b == NULL || beta == NULL || subgrad == NULL ||
      diag_H == NULL || I == NULL || ICPcounter == NULL || ICPs == NULL ||
      ACPs == NULL || cp_list == NULL || prevW == NULL || wt == NULL) {
    ppbmrm.exitflag = -2;
    goto cleanup;
  }

  map = (bool *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(bool));

  if (map == NULL) {
    ppbmrm.exitflag = -2;
    goto cleanup;
  }

  memset((bool *)map, true, BufSize);

  /* Temporary buffers for ICP removal */
  H_buff = (double *)LIBBMRM_CALLOC((size_t)BufSize * BufSize, sizeof(double));

  if (H_buff == NULL) {
    ppbmrm.exitflag = -2;
    goto cleanup;
  }

  /* Temporary buffers */
  beta_start = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  beta_good = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  b2 = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  diag_H2 = (double *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(double));

  H2 = (double *)LIBBMRM_CALLOC((size_t)BufSize * BufSize, sizeof(double));

  I_start = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  I_good = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  I2 = (int *)LIBBMRM_CALLOC((size_t)BufSize, sizeof(int));

  if (beta_start == NULL || beta_good == NULL || b2 == NULL ||
      diag_H2 == NULL || I_start == NULL || I_good == NULL || I2 == NULL ||
      H2 == NULL) {
    ppbmrm.exitflag = -2;
    goto cleanup;
  }

  ppbmrm.hist_Fp.resize(BufSize);
  ppbmrm.hist_Fd.resize(BufSize);
  ppbmrm.hist_wdist.resize(BufSize);

  /* Iinitial solution */
  R = (double)this->risk(W, subgrad);

  ppbmrm.nCP = 0;
  ppbmrm.nIter = 0;
  ppbmrm.exitflag = 0;

  b[0] = -R;

  /* Cutting plane auxiliary double linked list */
  LIBBMRM_MEMCPY(A, subgrad, (size_t)nDim * sizeof(double));
  map[0] = false;
  cp_list->address = &A[0];
  cp_list->idx = 0;
  cp_list->prev = NULL;
  cp_list->next = NULL;
  CPList_head = cp_list;
  CPList_tail = cp_list;

  /* Compute initial value of Fp, Fd, assuming that W is zero vector */
  sq_norm_W = 0.0;
  sq_norm_Wdiff = 0.0;

  for (int j = 0; j < nDim; ++j) {
    b[0] += subgrad[j] * W[j];
    sq_norm_W += W[j] * W[j];
    sq_norm_Wdiff += (W[j] - prevW[j]) * (W[j] - prevW[j]);
  }

  ppbmrm.Fp = R + 0.5 * _lambda * sq_norm_W + alpha * sq_norm_Wdiff;
  ppbmrm.Fd = -LIBBMRM_PLUS_INF;
  lastFp = ppbmrm.Fp;
  wdist = ::sqrt(sq_norm_Wdiff);

  // K = (sq_norm_W == 0.0) ? 0.4 : 0.01*::sqrt(sq_norm_W);

  LIBBMRM_MEMCPY(prevW, W, (size_t)nDim * sizeof(double));

  time(&tstop);

  /* Keep history of Fp, Fd, wdist */
  ppbmrm.hist_Fp[0] = ppbmrm.Fp;
  ppbmrm.hist_Fd[0] = ppbmrm.Fd;
  ppbmrm.hist_wdist[0] = wdist;

  /* Verbose output */
  if (verbose)
    printf("%4d: tim=%.3f, Fp=%f, Fd=%f, R=%f, K=%f\n", ppbmrm.nIter,
           difftime(tstop, tstart), ppbmrm.Fp, ppbmrm.Fd, R, K);

  /* main loop */

  while (ppbmrm.exitflag == 0) {
    time(&tstart);
    ppbmrm.nIter++;

    /* Update H */

    if (ppbmrm.nCP > 0) {
      A_2 = get_cutting_plane(CPList_tail);
      cp_ptr = CPList_head;

      for (int i = 0; i < ppbmrm.nCP; ++i) {
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;
        rsum = 0.0;

        for (int j = 0; j < nDim; ++j) {
          rsum += A_1[j] * A_2[j];
        }

        H[LIBBMRM_INDEX(i, ppbmrm.nCP, BufSize)] = rsum;
      }

      for (int i = 0; i < ppbmrm.nCP; ++i) {
        H[LIBBMRM_INDEX(ppbmrm.nCP, i, BufSize)] =
            H[LIBBMRM_INDEX(i, ppbmrm.nCP, BufSize)];
      }
    }

    rsum = 0.0;
    A_2 = get_cutting_plane(CPList_tail);

    for (int i = 0; i < nDim; ++i) rsum += A_2[i] * A_2[i];

    H[LIBBMRM_INDEX(ppbmrm.nCP, ppbmrm.nCP, BufSize)] = rsum;

    diag_H[ppbmrm.nCP] = H[LIBBMRM_INDEX(ppbmrm.nCP, ppbmrm.nCP, BufSize)];
    I[ppbmrm.nCP] = 1;

    ppbmrm.nCP++;
    beta[ppbmrm.nCP] = 0.0;  // [beta; 0]

    /* tune alpha cycle */
    /* ---------------------------------------------------------------------- */

    flag = true;
    isThereGoodSolution = false;
    LIBBMRM_MEMCPY(beta_start, beta, (size_t)ppbmrm.nCP * sizeof(double));
    LIBBMRM_MEMCPY(I_start, I, (size_t)ppbmrm.nCP * sizeof(int));
    qp_cnt = 0;
    alpha_good = alpha;

    if (tuneAlpha) {
      alpha_start = alpha;
      alpha = 0.0;
      beta[ppbmrm.nCP] = 0.0;
      LIBBMRM_MEMCPY(I2, I_start, (size_t)ppbmrm.nCP * sizeof(int));
      I2[ppbmrm.nCP] = 1;

      /* add alpha-dependent terms to H, diag_h and b */
      cp_ptr = CPList_head;

      for (int i = 0; i < ppbmrm.nCP; ++i) {
        rsum = 0.0;
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;

        for (int j = 0; j < nDim; ++j) rsum += A_1[j] * prevW[j];

        b2[i] = b[i] - ((2 * alpha) / (_lambda + 2 * alpha)) * rsum;
        diag_H2[i] = diag_H[i] / (_lambda + 2 * alpha);

        for (int j = 0; j < ppbmrm.nCP; ++j)
          H2[LIBBMRM_INDEX(i, j, BufSize)] =
              H[LIBBMRM_INDEX(i, j, BufSize)] / (_lambda + 2 * alpha);
      }

      /* solve QP with current alpha */
      qp_exitflag = libqp_splx_solver(&get_col, diag_H2, b2, &C, I2, &S, beta,
                                      ppbmrm.nCP, QPSolverMaxIter, 0.0,
                                      QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
      ppbmrm.qp_exitflag = qp_exitflag.exitflag;
      qp_cnt++;
      Fd_alpha0 = -qp_exitflag.QP;

      /* obtain w_t and check if norm(w_{t+1} -w_t) <= K */
      for (int i = 0; i < nDim; ++i) {
        rsum = 0.0;
        cp_ptr = CPList_head;

        for (int j = 0; j < ppbmrm.nCP; ++j) {
          A_1 = get_cutting_plane(cp_ptr);
          cp_ptr = cp_ptr->next;
          rsum += A_1[i] * beta[j];
        }

        wt[i] = (2 * alpha * prevW[i] - rsum) / (_lambda + 2 * alpha);
      }

      sq_norm_Wdiff = 0.0;

      for (int i = 0; i < nDim; ++i)
        sq_norm_Wdiff += (wt[i] - prevW[i]) * (wt[i] - prevW[i]);

      if (::sqrt(sq_norm_Wdiff) <= K) {
        flag = false;

        if (alpha != alpha_start) alphaChanged = true;
      } else {
        alpha = alpha_start;
      }

      while (flag) {
        LIBBMRM_MEMCPY(I2, I_start, (size_t)ppbmrm.nCP * sizeof(int));
        LIBBMRM_MEMCPY(beta, beta_start, (size_t)ppbmrm.nCP * sizeof(double));
        I2[ppbmrm.nCP] = 1;
        beta[ppbmrm.nCP] = 0.0;

        /* add alpha-dependent terms to H, diag_h and b */
        cp_ptr = CPList_head;

        for (int i = 0; i < ppbmrm.nCP; ++i) {
          rsum = 0.0;
          A_1 = get_cutting_plane(cp_ptr);
          cp_ptr = cp_ptr->next;

          for (int j = 0; j < nDim; ++j) rsum += A_1[j] * prevW[j];

          b2[i] = b[i] - ((2 * alpha) / (_lambda + 2 * alpha)) * rsum;
          diag_H2[i] = diag_H[i] / (_lambda + 2 * alpha);

          for (int j = 0; j < ppbmrm.nCP; ++j)
            H2[LIBBMRM_INDEX(i, j, BufSize)] =
                H[LIBBMRM_INDEX(i, j, BufSize)] / (_lambda + 2 * alpha);
        }

        /* solve QP with current alpha */
        qp_exitflag = libqp_splx_solver(&get_col, diag_H2, b2, &C, I2, &S, beta,
                                        ppbmrm.nCP, QPSolverMaxIter, 0.0,
                                        QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
        ppbmrm.qp_exitflag = qp_exitflag.exitflag;
        qp_cnt++;

        /* obtain w_t and check if norm(w_{t+1}-w_t) <= K */
        for (int i = 0; i < nDim; ++i) {
          rsum = 0.0;
          cp_ptr = CPList_head;

          for (int j = 0; j < ppbmrm.nCP; ++j) {
            A_1 = get_cutting_plane(cp_ptr);
            cp_ptr = cp_ptr->next;
            rsum += A_1[i] * beta[j];
          }

          wt[i] = (2 * alpha * prevW[i] - rsum) / (_lambda + 2 * alpha);
        }

        sq_norm_Wdiff = 0.0;
        for (int i = 0; i < nDim; ++i)
          sq_norm_Wdiff += (wt[i] - prevW[i]) * (wt[i] - prevW[i]);

        if (::sqrt(sq_norm_Wdiff) > K) {
          /* if there is a record of some good solution
           * (i.e. adjust alpha by division by 2) */

          if (isThereGoodSolution) {
            LIBBMRM_MEMCPY(beta, beta_good,
                           (size_t)ppbmrm.nCP * sizeof(double));
            LIBBMRM_MEMCPY(I2, I_good, (size_t)ppbmrm.nCP * sizeof(int));
            alpha = alpha_good;
            qp_exitflag = qp_exitflag_good;
            flag = false;
          } else {
            if (alpha == 0) {
              alpha = 1.0;
              alphaChanged = true;
            } else {
              alpha *= 2;
              alphaChanged = true;
            }
          }
        } else {
          if (alpha > 0) {
            /* keep good solution and try for alpha /= 2 if previous alpha was 1
             */
            LIBBMRM_MEMCPY(beta_good, beta,
                           (size_t)ppbmrm.nCP * sizeof(double));
            LIBBMRM_MEMCPY(I_good, I2, (size_t)ppbmrm.nCP * sizeof(int));
            alpha_good = alpha;
            qp_exitflag_good = qp_exitflag;
            isThereGoodSolution = true;

            if (alpha != 1.0) {
              alpha /= 2.0;
              alphaChanged = true;
            } else {
              alpha = 0.0;
              alphaChanged = true;
            }
          } else {
            flag = false;
          }
        }
      }
    } else {
      alphaChanged = false;
      LIBBMRM_MEMCPY(I2, I_start, (size_t)ppbmrm.nCP * sizeof(int));
      LIBBMRM_MEMCPY(beta, beta_start, (size_t)ppbmrm.nCP * sizeof(double));

      /* add alpha-dependent terms to H, diag_h and b */
      cp_ptr = CPList_head;

      for (int i = 0; i < ppbmrm.nCP; ++i) {
        rsum = 0.0;
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;

        for (int j = 0; j < nDim; ++j) rsum += A_1[j] * prevW[j];

        b2[i] = b[i] - ((2 * alpha) / (_lambda + 2 * alpha)) * rsum;
        diag_H2[i] = diag_H[i] / (_lambda + 2 * alpha);

        for (int j = 0; j < ppbmrm.nCP; ++j)
          H2[LIBBMRM_INDEX(i, j, BufSize)] =
              H[LIBBMRM_INDEX(i, j, BufSize)] / (_lambda + 2 * alpha);
      }
      /* solve QP with current alpha */
      qp_exitflag = libqp_splx_solver(&get_col, diag_H2, b2, &C, I2, &S, beta,
                                      ppbmrm.nCP, QPSolverMaxIter, 0.0,
                                      QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
      ppbmrm.qp_exitflag = qp_exitflag.exitflag;
      qp_cnt++;
    }

    /* -----------------------------------------------------------------------------------------------
     */

    /* Update ICPcounter (add one to unused and reset used) + compute number of
     * active CPs */
    ppbmrm.nzA = 0;

    for (int aaa = 0; aaa < ppbmrm.nCP; ++aaa) {
      if (beta[aaa] > epsilon) {
        ++ppbmrm.nzA;
        ICPcounter[aaa] = 0;
      } else {
        ICPcounter[aaa] += 1;
      }
    }

    /* W update */
    for (int i = 0; i < nDim; ++i) {
      rsum = 0.0;
      cp_ptr = CPList_head;

      for (int j = 0; j < ppbmrm.nCP; ++j) {
        A_1 = get_cutting_plane(cp_ptr);
        cp_ptr = cp_ptr->next;
        rsum += A_1[i] * beta[j];
      }

      W[i] = (2 * alpha * prevW[i] - rsum) / (_lambda + 2 * alpha);
    }

    /* risk and subgradient computation */
    R = (double)this->risk(W, subgrad);
    b[ppbmrm.nCP] = -R;
    add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize),
                      subgrad, nDim);

    sq_norm_W = 0.0;
    sq_norm_Wdiff = 0.0;
    sq_norm_prevW = 0.0;

    for (int j = 0; j < nDim; ++j) {
      b[ppbmrm.nCP] += subgrad[j] * W[j];
      sq_norm_W += W[j] * W[j];
      sq_norm_Wdiff += (W[j] - prevW[j]) * (W[j] - prevW[j]);
      sq_norm_prevW += prevW[j] * prevW[j];
    }

    /* compute Fp and Fd */
    ppbmrm.Fp = R + 0.5 * _lambda * sq_norm_W + alpha * sq_norm_Wdiff;
    ppbmrm.Fd = -qp_exitflag.QP +
                ((alpha * _lambda) / (_lambda + 2 * alpha)) * sq_norm_prevW;

    /* gamma + tuneAlpha flag */
    if (alphaChanged) {
      eps = 1.0 - (ppbmrm.Fd / ppbmrm.Fp);
      gamma = (lastFp * (1 - eps) - Fd_alpha0) / (Tmax * (1 - eps));
    }

    if ((lastFp - ppbmrm.Fp) <= gamma) {
      tuneAlpha = true;
    } else {
      tuneAlpha = false;
    }

    /* Stopping conditions - set only with nonzero alpha */
    if (alpha == 0.0) {
      if (ppbmrm.Fp - ppbmrm.Fd <= TolRel * LIBBMRM_ABS(ppbmrm.Fp))
        ppbmrm.exitflag = 1;

      if (ppbmrm.Fp - ppbmrm.Fd <= TolAbs) ppbmrm.exitflag = 2;
    }

    if (ppbmrm.nCP >= BufSize) ppbmrm.exitflag = -1;

    // tstop=ttime.cur_time_diff(false);
    time(&tstop);

    /* compute wdist (= || W_{t+1} - W_{t} || ) */
    sq_norm_Wdiff = 0.0;

    for (int i = 0; i < nDim; ++i) {
      sq_norm_Wdiff += (W[i] - prevW[i]) * (W[i] - prevW[i]);
    }

    wdist = ::sqrt(sq_norm_Wdiff);

    /* Realocate if there more cutting planes then buffer*/
    if (ppbmrm.hist_Fp.size() >= ppbmrm.nIter) {
      ppbmrm.hist_Fp.resize(int(1.5 * ppbmrm.nIter + 1));
      ppbmrm.hist_Fd.resize(int(1.5 * ppbmrm.nIter + 1));
      ppbmrm.hist_wdist.resize(int(1.5 * ppbmrm.nIter + 1));
    }

    /* Keep history of Fp, Fd, wdist */
    ppbmrm.hist_Fp[ppbmrm.nIter] = ppbmrm.Fp;
    ppbmrm.hist_Fd[ppbmrm.nIter] = ppbmrm.Fd;
    ppbmrm.hist_wdist[ppbmrm.nIter] = wdist;

    /* Verbose output */
    if (verbose)
      printf(
          "%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, "
          "nCP=%d, nzA=%d, wdist=%lf, alpha=%lf, qp_cnt=%d, gamma=%lf, "
          "tuneAlpha=%d\n",
          ppbmrm.nIter, difftime(tstop, tstart), ppbmrm.Fp, ppbmrm.Fd,
          ppbmrm.Fp - ppbmrm.Fd, (ppbmrm.Fp - ppbmrm.Fd) / ppbmrm.Fp, R,
          ppbmrm.nCP, ppbmrm.nzA, wdist, alpha, qp_cnt, gamma, tuneAlpha);

    /* Check size of Buffer */
    if (ppbmrm.nCP >= BufSize) {
      ppbmrm.exitflag = -2;
      printf("Buffer exceeded.\n");
    }

    /* keep w_t + Fp */
    LIBBMRM_MEMCPY(prevW, W, nDim * sizeof(double));
    lastFp = ppbmrm.Fp;

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
        nCP_new = ppbmrm.nCP - cntICP;

        for (int i = 0; i < cntICP; ++i) {
          tmp_idx = 0;
          cp_ptr = CPList_head;

          while (cp_ptr->address != ICPs[i]) {
            cp_ptr = cp_ptr->next;
            tmp_idx++;
          }

          remove_cutting_plane(&CPList_head, &CPList_tail, map, ICPs[i]);

          LIBBMRM_MEMMOVE(b + tmp_idx, b + tmp_idx + 1,
                          (ppbmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(beta + tmp_idx, beta + tmp_idx + 1,
                          (ppbmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(diag_H + tmp_idx, diag_H + tmp_idx + 1,
                          (ppbmrm.nCP - tmp_idx) * sizeof(double));
          LIBBMRM_MEMMOVE(I + tmp_idx, I + tmp_idx + 1,
                          (ppbmrm.nCP - tmp_idx) * sizeof(int));
          LIBBMRM_MEMMOVE(ICPcounter + tmp_idx, ICPcounter + tmp_idx + 1,
                          (ppbmrm.nCP - tmp_idx) * sizeof(int));
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

        ppbmrm.nCP = nCP_new;
      }
    }
  } /* end of main loop */

  ppbmrm.hist_Fp.resize(ppbmrm.nIter);
  ppbmrm.hist_Fd.resize(ppbmrm.nIter);
  ppbmrm.hist_wdist.resize(ppbmrm.nIter);

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
  LIBBMRM_FREE(A);
  LIBBMRM_FREE(subgrad);
  LIBBMRM_FREE(diag_H);
  LIBBMRM_FREE(I);
  LIBBMRM_FREE(ICPcounter);
  LIBBMRM_FREE(ICPs);
  LIBBMRM_FREE(ACPs);
  LIBBMRM_FREE(H_buff);
  LIBBMRM_FREE(map);
  LIBBMRM_FREE(prevW);
  LIBBMRM_FREE(wt);
  LIBBMRM_FREE(beta_start);
  LIBBMRM_FREE(beta_good);
  LIBBMRM_FREE(I_start);
  LIBBMRM_FREE(I_good);
  LIBBMRM_FREE(I2);
  LIBBMRM_FREE(b2);
  LIBBMRM_FREE(diag_H2);
  LIBBMRM_FREE(H2);

  if (cp_list) LIBBMRM_FREE(cp_list);

  return (ppbmrm);
}