#ifndef _MATRIX_IGM_HPP__
#define _MATRIX_IGM_HPP__

#include <array>
#include <valarray>
#include <cassert>
#include <numeric>
#include <omp.h>


namespace igm
{
  template<bool B, typename T = void>
  using Enable_if = typename std::enable_if<B, T>::type;

  template<typename List>
  bool check_non_jagged(const List& list)
  {
    auto i = list.begin();
    for (auto j = i + 1; j != list.end(); ++j)
      if (i->size() != j->size())
        return false;
    return true;
  }
  
  template<size_t N, typename I, typename List>
  Enable_if<(N>1), void> add_extents(I& first, const List& list)
  {
    assert(check_non_jagged(list));
    *first = list.size();
    add_extents<N - 1>(++first, *list.begin());
  }


  template<size_t N, typename I, typename List>
  Enable_if<(N == 1), void> add_extents(I& first, const List& list)
  {
    *(first++) = list.size(); // we reached the deepest nesting
  }


  template<size_t N, typename List>
  std::array<size_t, N> derive_extents(const List& list)
  {
    std::array<size_t, N> a;
    auto f = a.begin();
    add_extents<N>(f, list);
    return a;
  }

  template<typename T>
  class Mat {
  public:
    using vec_type = std::valarray<T>;
    using val_type = T;
    using idx_type = size_t;
    Mat() : Mat(0, 0) {}
    Mat(const Mat& M) : Mat(M._rows, M._cols) {
      _data = M._data;
      _slc = M._slc;
    }
    Mat(const size_t rows, const size_t cols, const T init = 0) : _rows{ rows }, _cols{ cols },
      _data(init, rows*cols), _slc{ 0,{ rows, cols },{ rows, 1 } },
      _nc{ _slc.size()[1] }, _nr{ _slc.size()[0] } {} // column major matrix

    void resize(const size_t rows, const size_t cols, const T init = 0)
    {
      _rows = rows;
      _cols = cols;
      _data.resize(rows*cols, init);
      _slc = std::gslice{ 0, { rows, cols },{ rows, 1 } };
      _nc = _slc.size()[1];
      _nr = _slc.size()[0];
    }
    Mat(std::initializer_list<std::initializer_list<T>> list);

    Mat& operator=(std::initializer_list<T>) = delete;
    Mat operator+(const Mat& A);
    void operator+=(const Mat& A);
    Mat operator-(const Mat& A);
    void operator-=(const Mat& A);
    Mat operator/(const Mat& A);
    void operator/=(const Mat& A);
    Mat operator*(const Mat& A);
    void operator*=(const Mat& A);
    Mat operator%(const Mat& A);
    void operator%=(const Mat& A);

    Mat operator+(const T s);
    void operator+=(T s) { _data += s; }
    Mat operator-(const T s);
    void operator-=(T s) { _data -= s; }
    Mat operator/(const T s);
    void operator/=(T s) { _data /= s; }
    Mat operator*(const T s);
    void operator*=(T s) { _data *= s; }
    Mat operator%(const T s);
    void operator%=(T s) { _data %= s; }

    vec_type& v() { return _data; }
    T* M() { return &_data[0]; }
    T* M(size_t r, size_t c) {
      return &_data[_slc.start() + lda()*c + r];
    }
    T* begin() { return std::begin(_data); }
    const T* begin() const { return std::begin(_data); }
    T* end() { return std::end(_data); }
    const T* end() const { return std::end(_data); }
    T* begincol(const size_t col) { return std::begin(_data) + (_slc.start() + lda()*col); }
    const T* begincol(const size_t col) const { return std::begin(_data) + (_slc.start() + lda()*col); }
    T* endcol(const size_t col) { return std::begin(_data) + (_slc.start() + lda()*col + rows()); }
    const T* endcol(const size_t col) const { return std::begin(_data) + (_slc.start() + lda()*col + rows()); }

    size_t rows() { return _slc.size()[0]; }
    size_t cols() { return _slc.size()[1]; }
    const size_t rows() const { return _slc.size()[0]; }
    const size_t cols() const { return _slc.size()[1]; }

    size_t size() { return _rows*_cols; }
    const size_t size() const { return _rows*_cols; }
    size_t lda() { return _rows; }
    const size_t lda() const { return _rows; }

    bool issub() {
      return _slc != std::gslice{ 0,{ _rows, _cols },{ _rows, 1 } };
    }
    void subreset() 
    { _slc = { 0,{ _rows, _cols },{ _rows, 1 } }; 
      _nc = _cols; _nr = _rows; }
    std::gslice slc() { return _slc; }
    const std::gslice slc() const { return _slc; }
    void sub(std::gslice slc) { _slc = slc; }
    Mat<T> sub(Mat<size_t>& idx);
    void subcols(Mat& A, Mat<size_t>& idx);
    void sub(const size_t rFirst, const size_t rLast,
      const size_t cFirst, const size_t cLast);
    void subcols(const size_t first, const size_t last);
    void subcols(const size_t first);
    Mat<T>& subcol(const size_t col);
    void subrow(const size_t col);

    void swapcols(const size_t c1, const size_t c2);

    bool empty() { return _rows == 0 || _cols == 0; }


    T max(size_t& idx);
    void fill(const T val) { _data[_slc] = val; }
    void zeros() { _data[_slc] = T{ 0 }; }
    void iota(const T start) { std::iota(std::begin(_data), std::end(_data), start); }
    T& operator()(size_t r, size_t c) {
      return _data[_slc.start() + lda()*c + r];
    }
    const T& operator()(size_t r, size_t c) const {
      //std::cout << "*idx " << (_slc.start() + lda()*c + r) << " of " << _data.size() << "\n";
      return _data[_slc.start() + lda()*c + r];
    }
    T& operator()(size_t idx) {
      //std::cout << "*idx " << _slc.start() + lda()*idx << " of " << _data.size() << "\n";
      return _data[_slc.start() + lda()*idx];
    }
    const T& operator()(size_t idx) const {
      return _data[_slc.start() + lda()*idx];
    }

    T at(const size_t idx) {
      return _data[_slc.start() + idx];
    }
    const T at(const size_t idx) const {
      return _data[_slc.start() + idx];
    }

    // misc algorithms
    void eye()
    {
      if (_nc != _nr)
        throw std::exception("Invalid dimensions in eye!");
      _data = 0;
      _data[std::slice(0, _nr, _nc + 1)] = T{ 1 };
    }

    // output
    void print(const char* str);

  protected:
    size_t _rows = 0;
    size_t _cols = 0;
    vec_type _data;
    std::gslice _slc;

  public:
    size_t _nr;
    size_t _nc;
  };


  template<typename T>
  std::ostream& operator<<(std::ostream& os, Mat<T>& m)
  {
    os << "matrix[" << m.rows() << "," << m.cols() << "]\n";
    for (size_t i = 0; i < m.rows(); ++i)
    {
      for (size_t j = 0; j < m.cols(); ++j)
        os << m(i, j) << " ";
      os << "\n";
    }
    return os;
  }

  template<typename T>
  inline Mat<T>::Mat(std::initializer_list<std::initializer_list<T>> list)
  {
    auto ext = derive_extents<2>(list);
    _nr = _rows = ext[1];
    _nc = _cols = ext[0];
    _slc = { 0,{ _rows, _cols },{ 1, _rows } };
    _data.resize(_rows*_cols);
    for (size_t i = 0; i < _cols; ++i)
    {
      auto nl = list.begin() + i;
      for (size_t j = 0; j < _rows; ++j)
      {
        auto v = nl->begin();
        _data[_rows*i + j] = *(v + j);
      }
    }
  }


  template<typename T>
  Mat<T> Mat<T>::operator+(const Mat & A)
  {
    Mat<T> B(A);
    B._data[B._slc] += _data[_slc];

    return B;
  }


  template<typename T>
  void Mat<T>::operator+=(const Mat & A)
  {
    _data[_slc] += A._data[A._slc];
  }


  template<typename T>
  Mat<T> Mat<T>::operator-(const Mat & A)
  {
    Mat<T> B(A);
    B._data[B._slc] -= _data[_slc];

    return B;
  }

  template<typename T>
  void Mat<T>::operator-=(const Mat & A)
  {
    _data[_slc] -= A._data[A._slc];
  }


  template<typename T>
  Mat<T> Mat<T>::operator/(const Mat & A)
  {
    Mat<T> B(A);
    B._data[B._slc] /= _data[_slc];

    return B;
  }

  template<typename T>
  void Mat<T>::operator/=(const Mat & A)
  {
    _data[_slc] /= A._data[A._slc];
  }


  template<typename T>
  Mat<T> Mat<T>::operator%(const Mat & A)
  {
    Mat<T> B(A);
    B._data[B._slc] %= _data[_slc];

    return B;
  }

  template<typename T>
  void Mat<T>::operator%=(const Mat & A)
  {
    _data[_slc] %= A._data[A._slc];
  }

  template<typename T>
  Mat<T> Mat<T>::operator*(const Mat & A)
  {
    Mat<T> B(A);
    B._data[B._slc] *= _data[_slc];

    return B;
  }

  template<typename T>
  void Mat<T>::operator*=(const Mat & A)
  {
    _data[_slc] *= A._data[A._slc];
  }



  template<typename T>
  void Mat<T>::sub(const size_t rFirst, const size_t rLast, 
    const size_t cFirst, const size_t cLast)
  {
    auto rows = rLast - rFirst + 1;
    auto cols = cLast - cFirst + 1;
    _slc = std::gslice{ cFirst*lda() + rFirst,
      { rows, cols }, { 1, lda() } };
    _nc = _slc.size()[1];
    _nr = _slc.size()[0];
  }



  template<typename T>
  void Mat<T>::subcols(const size_t first, const size_t last)
  {
    sub(0, _rows - 1, first, last);
  }

  template<typename T>
  void Mat<T>::subcols(const size_t first)
  {
    sub(0, _rows - 1, first, _cols - 1);
  }

  template<typename T>
  inline Mat<T>& Mat<T>::subcol(const size_t col)
  {
    sub(0, _rows - 1, col, col);
    return *this;
  }

  template<typename T>
  inline void Mat<T>::subrow(const size_t row)
  {
    sub(row, row, 0, _cols - 1);
  }

  template<typename T>
  void Mat<T>::swapcols(const size_t c1, const size_t c2)
  {
    std::swap_ranges(begincol(c1), endcol(c1), begincol(c2));
  }

  template<typename T>
  inline T Mat<T>::max(size_t & idx)
  {
    T max = at(0);
    idx = 0;
    for (size_t i = 0; i < _nc; ++i)
    {
      if (at(i) > max) {
        idx = i;
        max = at(i);
      }
    }
    return max;
  }

  template<typename T>
  void Mat<T>::print(const char * str)
  {
    std::cout << str << *this;
  }

  template<typename T>
  Mat<T> Mat<T>::sub(Mat<size_t>& idx)
  {
    Mat<T> A(_nr, idx._nc);

    for (size_t i = 0; i < idx._nc; ++i) {
      std::memcpy(A.begincol(i), begincol(idx(0,i)), sizeof(T)*_nr);
    }

    return A;
  }

  template<typename T>
  inline void Mat<T>::subcols(Mat& A, Mat<size_t>& idx)
  {
    for (size_t i = 0; i < idx._nc; ++i) {
      std::memcpy(A.begincol(i), begincol(idx(0, i)), sizeof(T)*_nr);
    }
  }


  template<typename T>
  T sum(const Mat<T>& v)
  {
    //if (v.issub())
    //  throw std::exception("sum doesnt't operate on sub-views!");
    return std::accumulate(v.begin(), v.end(), T{ 0 });
  }

  template<typename T>
  T sumabs2_col1(const Mat<T>& src, const size_t col)
  {
//#define TRY_PARALLEL
#ifdef TRY_PARALLEL
    int nelements = src._nr;
    T* a = const_cast<double*>(src.begincol(col));
    T* b = a;
    T sumt{ 0 };
    int nthreads;


#pragma omp parallel 
#pragma omp single
    nthreads = omp_get_num_threads();
    std::cout << nthreads << "\n";

#pragma omp parallel default(none) reduction(+:sumt) shared(a, b, nthreads, nelements) 
    {
      int tid = omp_get_thread_num();
      int nitems = nelements / nthreads;
      int start = tid * nitems;
      int end = start + nitems;
      if (tid == nthreads - 1) end = nelements;

      sumt += std::inner_product(a + start, a + end, b + start, 0);
    }
    return sumt;
#else
    return std::inner_product(src.begincol(col), src.endcol(col), src.begincol(col), T{ 0 });
#endif
  }

  template<typename T>
  void sumabs2_col(Mat<T>& dst, const Mat<T>& src, const size_t first)
  {
    if (dst.cols() != src.cols())
      throw std::exception("Invalid dimensions in sumabs2_col");
    for (size_t i = first; i < first+src._nc; ++i)
    {
      dst(0, i) = sumabs2_col1(src, i);
    }
  }

  template<typename T>
  void sumabs2_col(Mat<T>& dst, const Mat<T>& src, const size_t first,
     const Mat<T>& v)
  {
    if (dst.cols() != src.cols())
      throw std::exception("Invalid dimensions in sumabs2_col");
#pragma omp parallel for
    for (long long i = first; i < static_cast<long long>(src.cols()); ++i)
    {
      dst(0, i) = sumabs2_col1(src, i) + v(i);
    }
  }

  template<typename T>
  Mat<T> sumabs2_col(const Mat<T>& src, const size_t first)
  {
    Mat<T> dst(1, src._nc);
    sumabs2_col(dst, src, first);

    return dst;
  }

  template<typename T>
  T sumabs2(const Mat<T>& src)
  {
    return std::inner_product(src.begin(), src.end(), src.begin(), T{ 0 });
  }


  template<typename T>
  Mat<T> eye(const size_t rc)
  {
    Mat<T> A(rc, rc);
    A.eye();

    return A;
  }

}

#endif //_MATRIX_IGM_HPP__

