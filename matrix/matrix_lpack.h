#ifndef _MATRIX_LPACK_H__
#define _MATRIX_LPACK_H__


namespace igm {

  // solves upper triangular A*x=b 
  template<typename T>
  void solve(Mat<T>& x, const Mat<T>& A, const Mat<T>& b)
  {
    //size_t n = A._nr;
    size_t n = x._nc;
    for (size_t k = 0; k < n; ++k)
    {
      size_t i = n - 1 - k;
      T s{ 0 };
      for (size_t j = i + 1; j < n; ++j)
      {
        s = s + A(i, j) * x(0, j);
      }
      x(0, i) = (b(0, i) - s) / A(i, i);
    }
  }

  namespace dpr { // deprecated

    template<typename T>
    void mul3div_e(Mat<T>& dst, const Mat<T>& u, const Mat<T>& v,
      const Mat<T>& w, T a)
    {
      for (size_t i = 0; i < dst._nc; ++i)
      {
        dst(i) = u(i) * v(i) * w(i) * a;
      }
    }


    template<typename T>
    void subtmul2_e(Mat<T>& dst, const Mat<T>& u,
      const Mat<T>& v, const T a)
    {
      for (size_t i = 0; i < dst._nc; ++i)
      {
        dst(i) = u(i) - v(i, 0) * a;
      }
    }


    template<typename T>
    void mgs(Mat<T>& Q, Mat<T>& R)
    {
      for (size_t k = 0; k < Q._nc; ++k)
      {
        auto alpha = T{ 1 } / sumabs2_col1(Q, k);
        for (size_t j = k + 1; j < Q._nc; ++j)
        {
          T s{ 0 };
          for (size_t i = 0; i < Q._nr; ++i)
          {
            s += Q(i, k) * Q(i, j);
          }
          R(k, j) = s * alpha;
        }

        for (size_t j = k + 1; j < Q._nc; ++j)
        {
          for (size_t i = 0; i < Q._nr; ++i)
          {
            Q(i, j) -= Q(i, k) * R(k, j);
          }
        }
      }
    }


    template<typename T>
    void mgs_k(Mat<T>& Q, Mat<T>& R, const size_t k)
    {
      auto alpha = T{ 1 } / sumabs2_col1(Q, k);
      for (size_t j = k + 1; j < Q._nc; ++j)
      {
        T s{ 0 };
        for (size_t i = 0; i < Q._nr; ++i)
        {
          s += Q(i, k) * Q(i, j);
        }
        R(k, j) = s * alpha;
      }

      for (size_t j = k + 1; j < Q._nc; ++j)
      {
        for (size_t i = 0; i < Q._nr; ++i)
        {
          Q(i, j) -= Q(i, k) * R(k, j);
        }
      }
    }

    template<typename T>
    void mtv(Mat<T>& dst, const Mat<T>& A, const Mat<T>& v)
    {
      if (A.rows() != v.cols() || dst.cols() != A.cols())
        throw std::exception("Invalid dimensions in mtv");
#pragma omp parallel for
      for (long long i = 0; i < static_cast<long long>(dst.cols()); ++i)
      {
        dst(i) = std::inner_product(A.begincol(i), A.endcol(i), v.begin(), T{ 0 });
      }
    }

    // special A'*x, where x=A.row(0)
    template<typename T>
    void mtv_s(Mat<T>& dst, const Mat<T>& A, const T a)
    {
      for (size_t j = 1; j < dst._nc + 1; ++j)
      {
        T s{ 0 };
        for (size_t i = 0; i < A._nr; ++i)
        {
          s += A(i, j) * A(i, 0);
        }
        //throw "next line has undefined behaviour! revise!";
        //dst(0, j-1) = s * a;
        dst(j - 1) = s * a;
      }
    }


    // special A'*x, where x=A.row(0)
    template<typename T>
    void ger_s(Mat<T>& A, const Mat<T>& x, const T a)
    {
      for (size_t j = 1; j < x._nc; ++j)
      {
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(A._nr); ++i)
        {
          A(i, j) = A(i, j) + a * A(i, 0) * x(0, j);
        }
      }
    }


    //  element-wise dst = u / (v + w)
    template<typename T>
    void div_add(Mat<T>& dst, const Mat<T>& u,
      const Mat<T>& v, const Mat<T>& w)
    {
      for (size_t i = 0; i < dst._nc; ++i)
      {
        dst(i) = u(i) / (v(i) + w(i));
      }
    }
  }

} // namespace igm

#endif // _MATRIX_LPACK_H__
