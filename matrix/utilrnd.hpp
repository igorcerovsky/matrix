#ifndef _UTILRND_H__
#define _UTILRND_H__

#include <type_traits>
#include <random>
#include <functional>
#include <chrono>


template<typename T>
class RandReal
{
public:
  RandReal(T low, T high) : dist{ low, high } {};
  T operator()() { return r(); }
protected:
  std::default_random_engine re;
  std::uniform_real_distribution<T> dist;
  std::function<T()> r = std::bind(dist, re);
};

#endif // _UTILRND_H__