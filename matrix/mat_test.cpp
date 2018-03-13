// mat_test.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include "matrix_igm.hpp"
#include "matrix_lpack.h"
#include "matrix_lpack_blas.h"


int main()
{
  using Mat = igm::Mat<double>;
  Mat Q = { { 7., 3., 3., 8., 2., 4. },{ 6., 9., 4., 8., 1., 5. },{ 4., 3., 6., 3., 4., 1. },
  { 1., 5., 9., 1., 7., 9. },{ 5., 6., 2., 8., 3., 1. } };
  Mat R = { { 1., 0., 0., 0., 0. },
  { 0., 1., 0., 0., 0. },
  { 0., 0., 1., 0., 0. },
  { 0., 0., 0., 1., 0. },
  { 0., 0., 0., 0., 1. },
  };

  size_t l{ 0 };
  R.sub(l, l, l + 1, R._nc - 1);
  R.print("R: ");

  Q.subcols(l + 1);
  Q.print("Q: ");

  igm::blas::gemv(R, Q);
  R.print("R_gemv: ");
  Q.print("Q_gemv: ");

  R.subreset();
  R.print("R_gemv: ");
  Q.subreset();
  Q.print("Q_gemv: ");

  return 0;
}

