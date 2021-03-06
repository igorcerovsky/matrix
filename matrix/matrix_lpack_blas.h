#ifndef _MATRIX_LPACK_BLAS_H__
#define _MATRIX_LPACK_BLAS_H__

#include "mkl.h"

namespace igm {

  using blasint = MKL_INT;


  namespace blas {
    template<typename T>
    void ger(Mat<T>& A, const Mat<T>& x, const Mat<T>& y, const T a = T{1})
    {
      CBLAS_ORDER order = CblasColMajor;
      double* a_ = A.begin();
      const double* x_ = x.begin();
      const double* y_ = y.begin();
      blasint m = static_cast<blasint>(A._nr);
      blasint n = static_cast<blasint>(A._nc);
      blasint incx = static_cast<blasint>(x.lda());
      blasint incy = static_cast<blasint>(y.lda());
      blasint lda = static_cast<blasint>(A.lda());
      cblas_dger(order, m, n, a,
        x_, 1, y_, 1,
        a_, lda);
    }


    template<typename T>
    void gemv(Mat<T>& y, const Mat<T>& A, const Mat<T>& x,
      const T a = T{ 1 }, const T b = T{ 0 }, CBLAS_TRANSPOSE transa = CblasTrans)
    {
      CBLAS_ORDER order = CblasColMajor;
      const double* a_ = A.begin();
      const double* x_ = x.begin();
      double* y_ = y.begin();
      blasint m = static_cast<blasint>(A._nr);
      blasint n = static_cast<blasint>(A._nc);
      blasint incx = static_cast<blasint>(x.lda());
      blasint incy = static_cast<blasint>(y.lda());
      cblas_dgemv(order, transa, m, n, a,
        a_, m, x_, incx, b,
        y_, incy);
    }


    template<typename T>
    void ger(Mat<T>& A, const Mat<T>& y, const T a = T{ 1 })
    {
      CBLAS_ORDER order = CblasColMajor;
      double* a_ = A.begin() + A.slc().start();
      const double* x_ = a_ - A.lda();
      const double* y_ = y.begin() + y.slc().start();
      blasint m = static_cast<blasint>(A._nr);
      blasint n = static_cast<blasint>(A._nc);
      blasint incx = 1; //static_cast<blasint>(A.lda());
      blasint incy = static_cast<blasint>(y.lda());
      blasint lda = static_cast<blasint>(A.lda());
      cblas_dger(order, m, n, a,
        x_, incx, y_, incy,
        a_, lda);
    }


    template<typename T>
    void gemv(Mat<T>& y, const Mat<T>& Q, const T a = T{ 1 })
    {
      CBLAS_ORDER order = CblasColMajor;
      CBLAS_TRANSPOSE transa = CblasTrans;
      const double* a_ = Q.begin() + Q.slc().start();
      const double* x_ = a_ - Q.lda();
      double* y_ = y.begin() + y.slc().start();
      blasint m = static_cast<blasint>(Q._nr);
      blasint n = static_cast<blasint>(Q._nc);
      blasint incx = 1;
      blasint incy = static_cast<blasint>(y.lda());
      cblas_dgemv(order, transa, m, n, a,
        a_, m, x_, incx, 0,
        y_, incy);
    }


  }

} // namespace igm

#endif // _MATRIX_LPACK_BLAS_H__
