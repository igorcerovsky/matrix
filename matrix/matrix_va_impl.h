#ifndef _MATRIX_IMPL_H__
#define _MATRIX_IMPL_H__

#include <algorithm>
#include <numeric>

namespace tim {

  template<typename T>
  inline T dot(const std::valarray<T>& v, const std::valarray<T>& u)
  {
    return std::inner_product(&v[0], &v[0] + v.size(), &u[0], T{ 0 });
  }

  template<typename T>
  T len(const std::valarray<T>& v)
  {
    return sqrt(dot(v, v));
  }

  template<typename T>
  void unit(std::valarray<T>& u)
  {
    u *= T{ 1 } / len<T>(u);
  }


  template<typename T>
  std::valarray<T> unit(const std::valarray<T>& u)
  {
    std::valarray<T> r = u;
    unit(r);
    return r;
  }

} // namespace mtrx

#endif // _MATRIX_IMPL_H__
