// unit_test.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <numeric>
#include "gtest.h"

#include "../matrix/matrix_igm.hpp"
#include "../matrix/matrix_lpack.h"
#include "../matrix/matrix_lpack_blas.h"


#define SHOW_RESULTS

using MatI = igm::Mat<int>;
using MatD = igm::Mat<double>;
using std::cout;

template<typename T>
void Iota(std::valarray<T>& v, const T init=1)
{
  std::iota<T*, T>(std::begin(v), std::end(v), init);
}

// column order matrix
// data shall be sequentialy aligned in column order
TEST(matrix_constuctor_rows_cols, matrix_constuctor_rows_cols_init)
{
  MatI A(2, 3);
  Iota(A.v());
  cout << A;

  MatI B(3, 2);
  Iota(B.v());
  cout << B;

  MatI C(3, 1);
  Iota(C.v());
  cout << C;

  MatI D(1, 3);
  Iota(D.v());
  cout << D;
}


// column order matrix
// data shall be sequentialy aligned in column order
TEST(matrix_initializer_list_constuctor, matrix_initializer_list_constuctor_index_order)
{
  MatI A{ { 1, 2, 3 },{ 4, 5, 6 } };
  MatI::val_type* pData = A.M();
  for (size_t i = 0; i < A.size(); ++i)
  {
    ASSERT_EQ(pData[i], i + 1);
  }
}


TEST(matrix_operator_plus, matrix_operator_plus_no_slice)
{
  MatI A{ { 6, 5, 4 }, { 3, 2, 1 } };
  MatI B{ { 1, 2, 3 }, { 4, 5, 6 } };
  MatI C = A + B;
  std::cout << A << "\n";
  std::cout << B << "\n";
  std::cout << C << "\n";
  MatI R{ { 7, 7, 7 },{ 7, 7, 7 } };
  MatI::val_type* pData = A.M();
  for (size_t i = 0; i < A.size(); ++i)
  {
    ASSERT_EQ(*(C.M()+i), *(R.M()));
  }
}


TEST(matrix_slice_test, matrix_slice_test_columns)
{
  MatI A{ { 1,   2,  3,  4,  5 },
          { 6,   7,  8,  9, 10 },
          { 11, 12, 13, 14, 15 },
          { 16, 17, 18, 19, 20 } };
  
  std::cout << A << "\n";

  A.subcols(1, 2);
  std::cout << A << "\n";
  ASSERT_EQ(A(0, 0), 6);   ASSERT_EQ(A(0, 1), 11);
  ASSERT_EQ(A(1, 0), 7);   ASSERT_EQ(A(1, 1), 12);
  ASSERT_EQ(A(2, 0), 8);   ASSERT_EQ(A(2, 1), 13);
  ASSERT_EQ(A(3, 0), 9);   ASSERT_EQ(A(3, 1), 14);
  ASSERT_EQ(A(4, 0), 10);  ASSERT_EQ(A(4, 1), 15);

  A.subcols(0, 0);
  std::cout << A << "\n";
  ASSERT_EQ(A(0, 0), 1);
  ASSERT_EQ(A(1, 0), 2);
  ASSERT_EQ(A(2, 0), 3);
  ASSERT_EQ(A(3, 0), 4);
  ASSERT_EQ(A(4, 0), 5);

  A.subcols(3, 3);
  std::cout << A << "\n";
  ASSERT_EQ(A(0, 0), 16);
  ASSERT_EQ(A(1, 0), 17);
  ASSERT_EQ(A(2, 0), 18);
  ASSERT_EQ(A(3, 0), 19);
  ASSERT_EQ(A(4, 0), 20);
}


TEST(matrix_slice_test, matrix_slice_test_columns_from)
{
  MatI A{ { 1,   2,  3,  4,  5 },
  { 6,   7,  8,  9, 10 },
  { 11, 12, 13, 14, 15 },
  { 16, 17, 18, 19, 20 } };

  std::cout << A << "\n";

  A.subcols(2);
  std::cout << A << "\n";
  ASSERT_EQ(A(0, 0), 11);  ASSERT_EQ(A(0, 1), 16);
  ASSERT_EQ(A(1, 0), 12);  ASSERT_EQ(A(1, 1), 17);
  ASSERT_EQ(A(2, 0), 13);  ASSERT_EQ(A(2, 1), 18);
  ASSERT_EQ(A(3, 0), 14);  ASSERT_EQ(A(3, 1), 19);
  ASSERT_EQ(A(4, 0), 15);  ASSERT_EQ(A(4, 1), 20);
}

TEST(matrix_row_vector_test, matrix_row_vector_test_init)
{
  MatI A{ { 1 },{ 2 },{ 3 },{ 4 },{ 5 } };
  std::cout << A << "\n";

  MatI B(1, 10);
  std::cout << B << "\n";
}


TEST(matrix_iterator_col_test, matrix_iterator_col_test_begin_end)
{
  MatI A{ { 1,   2,  3,  4,  5 },
  { 6,   7,  8,  9, 10 },
  { 11, 12, 13, 14, 15 },
  { 16, 17, 18, 19, 20 } };
  std::cout << A << "\n";

  auto itB = A.begincol(0);
  auto itE = A.endcol(0);
  int i = 1;
  for (auto a = itB; a != itE; ++a)
  {
    std::cout << *a << " ";
    ASSERT_EQ(*a, i);
    ++i;
  }
  std::cout << "\n";

  itB = A.begincol(3);
  itE = A.endcol(3);
  i = 16;
  for (auto a = itB; a != itE; ++a)
  {
    std::cout << *a << " ";
    ASSERT_EQ(*a, i);
    ++i;
  }
  std::cout << "\n";
}


TEST(matrix_iterator_col_test, matrix_iterator_col_test_slice)
{
  MatI A{ { 1,   2,  3,  4,  5 },
  { 6,   7,  8,  9, 10 },
  { 11, 12, 13, 14, 15 },
  { 16, 17, 18, 19, 20 } };
  std::cout << A << "\n";

  A.sub(1, 3, 1, 2);
  std::cout << A << "\n";

  auto itB = A.begincol(0);
  auto itE = A.endcol(0);
  int i = 7;
  for (auto a = itB; a != itE; ++a)
  {
    std::cout << *a << " ";
    ASSERT_EQ(*a, i);
    ++i;
  }
  std::cout << "\n";

  itB = A.begincol(1);
  itE = A.endcol(1);
  i = 12;
  for (auto a = itB; a != itE; ++a)
  {
    //std::cout << *a << " ";
    ASSERT_EQ(*a, i);
    ++i;
  }
  std::cout << "\n";
}


TEST(sumabs2_columns, sumabs2_columns_iter)
{
  MatI A{ { 1,   2,  3,  4,  5 },
  { 6,   7,  8,  9, 10 },
  { 11, 12, 13, 14, 15 },
  { 16, 17, 18, 19, 20 } };
  cout << A << "\n";

  MatI x(1, A.cols());
  igm::sumabs2_col(x, A, 0);
  cout << x << "\n";
  ASSERT_EQ(x(0, 0), 55);
  ASSERT_EQ(x(0, 1), 330);
  ASSERT_EQ(x(0, 2), 855);
  ASSERT_EQ(x(0, 3), 1630);

  A.sub(1, 2, 1, 3);
  MatI y(1, A._nc);
  igm::sumabs2_col(y, A, 0);
  cout << A << "\n";
  cout << y << "\n";
  ASSERT_EQ(y(0, 0), 113);
  ASSERT_EQ(y(0, 1), 313);
  ASSERT_EQ(y(0, 2), 613);
}


TEST(matrix_operator_plus, matrix_operator_plus_1)
{
  MatI A{ { 1,   2,  3,  4,  5 },
  { 6,   7,  8,  9, 10 },
  { 11, 12, 13, 14, 15 },
  { 16, 17, 18, 19, 20 } };
  cout << A << "\n";

  MatI B{ { 20,   19,  18,  17,  16 },
  { 15,   14,  13,  12, 11 },
  { 10, 9, 8, 7, 6 },
  { 5, 4, 3, 2, 1 } };
  cout << B << "\n";

  MatI C = A + B;
  cout << C << "\n";
  for (auto a = C.begin(); a != C.end(); ++a)
  {
    ASSERT_EQ(*a, 21);
  }
  A.sub(1, 2, 1, 3);
  B.sub(A.slc());
  cout << B << "\n";

  MatI D = A + B;
  cout << D << "\n";
  D.subreset();
  cout << D << "\n";
}

TEST(matrix_sub, matrix_sub_ids)
{
  MatI A{
    { 1,   2,  3,  4,  5 },
    { 6,   7,  8,  9, 10 },
    { 11, 12, 13, 14, 15 },
    { 16, 17, 18, 19, 20 } };
  cout << "A: " << A << "\n";
  igm::Mat<size_t> idx{ {1}, {3}, {0}, {2} };
  cout << "idx: " << idx << "\n";

  MatI C = A.sub(idx);
  cout << "C: " << C << "\n";

  MatI D(A._nr, A._nc);
  A.subcols(D, idx);
  cout << "D: " << D << "\n";
}


TEST(matrix_eye, matrix_eye_square)
{
  MatI A = igm::eye<MatI::val_type>(4);
  cout << A << "\n";
  for (size_t i = 0; i < A._nr; ++i)
  {
    ASSERT_EQ(A(i, i), 1);
  }
}

TEST(mtv, mtv_)
{
  MatI A{ { 1, 2, 6, 3 },{ 5, 4, 1, 7 },{ 1, 0, 3, 8 } };
  MatI x{ { 3}, {1}, {4}, {2} };
  A.print("A:");
  x.print("x:");
  MatI b(1, 3);

  igm::dpr::mtv(b, A, x);
  b.print("b:");
  ASSERT_EQ(b(0), 35);
  ASSERT_EQ(b(1), 37);
  ASSERT_EQ(b(2), 31);
}


TEST(mtv_s, smtv_1)
{
  MatI A{ { 3 , 1 , 4 , 2 },{ 1, 2, 6, 3 },{ 5, 4, 1, 7 },{ 1, 0, 3, 8 } };
  A.print("A:");
  MatI b(1, 3);

  igm::dpr::mtv_s(b, A, 1);
  b.print("b:");
  ASSERT_EQ(b(0), 35);
  ASSERT_EQ(b(1), 37);
  ASSERT_EQ(b(2), 31);
}


TEST(smtve, smtve_2)
{
  MatI A{ { 0, 0, 0, 0 },{ 3 , 1 , 4 , 2 },{ 1, 2, 6, 3 },{ 5, 4, 1, 7 },{ 1, 0, 3, 8 } };
  A.print("A:");
  MatI b(1, A._nr);

  constexpr size_t l = 1;
  A.subcols(l);
  b.subcols(l);
  A.print("A_sub:");
  igm::dpr::mtv_s(b, A, 1);
  b.print("b_sub:");
  b.subreset();
  b.print("b_sub:");
  ASSERT_EQ(b(0), 0);
  ASSERT_EQ(b(1), 35);
  ASSERT_EQ(b(2), 37);
  ASSERT_EQ(b(3), 31);
}


TEST(smtve, smtve_3)
{
  MatI A{ { 0, 0, 0, 0 }, { 3 , 1 , 4 , 2 }, { 1, 2, 6, 3 }, 
  { 5, 4, 1, 7 }, { 1, 0, 3, 8 } };
  MatI R(A._nr, A._nr);
  A.print("A:");
  R.print("R:");

  constexpr size_t l = 1;
  constexpr size_t r = 1;
  A.subcols(l);
  R.sub(r, r, l, R._nc-1);
  A.print("A_sub:");
  igm::dpr::mtv_s(R, A, 1);
  R.print("b_sub:");
  R.subreset();
  R.print("b_sub:");
  ASSERT_EQ(R(r, 0), 0);
  ASSERT_EQ(R(r, 1), 35);
  ASSERT_EQ(R(r, 2), 37);
  ASSERT_EQ(R(r, 3), 31);
}

TEST(ger_blas, ger_blas_1)
{
  MatD A = { { 7., 3., 3., 8. },{ 6., 9., 4., 8. },{ 4.,  3., 6., 3. } };
  MatD x = { { 1., 5., 7., 3. } };
  MatD y = { { 4., 9., 7. } };

  igm::blas::ger(A, x, y, 1.0);

  MatD R = { { 11., 23., 31., 20. },{ 15., 54., 67., 35. },{ 11., 38., 55., 24. } };
  for (size_t i = 0; i < R.size(); ++i)
  {
    ASSERT_EQ(R.at(i), A.at(i));
  }
}


TEST(ger_blas, ger_blas_special_case_1)
{
  MatD A = { { 1., 5., 7., 3. },{ 7., 3., 3., 8. },
  { 6., 9., 4., 8. },{ 4., 3., 6., 3. } };
  MatD y = { { 4. },{ 9. },{ 7. } };
  MatD R = { { 1., 5., 7., 3. },{ 11., 23., 31., 20. },
  { 15., 54., 67., 35. },{ 11., 38., 55., 24. } };

  size_t l{ 0 };
  A.subcols(l + 1);
  igm::blas::ger(A, y, 1.0);

  A.subreset();
  //A.print("A: ");
  //R.print("R: ");

  for (size_t i = 0; i < R.size(); ++i)
  {
    ASSERT_EQ(R.at(i), A.at(i));
  }
}


TEST(ger_blas, ger_blas_special_case_2)
{
  MatD A = { { 0., 0., 0., 0. }, { 1., 5., 7., 3. }, { 7., 3., 3., 8. },
  { 6., 9., 4., 8. },{ 4., 3., 6., 3. } };
  MatD y = { { 4. },{ 9. },{ 7. } };
  MatD R = { { 0., 0., 0., 0. }, { 1., 5., 7., 3. }, { 11., 23., 31., 20. },
  { 15., 54., 67., 35. }, { 11., 38., 55., 24. } };

  size_t l{ 1 };
  A.subcols(l + 1);
  igm::blas::ger(A, y, 1.0);

  A.subreset();
  A.print("A: ");
  R.print("R: ");

  for (size_t i = 0; i < R.size(); ++i)
  {
    ASSERT_EQ(R.at(i), A.at(i));
  }
}


TEST(gemv_blas, gemv_blas_1)
{
  MatD A = { { 6., 9., 4., 8., 1., 5. },{ 4., 3., 6., 3., 4., 1. },
  { 1., 5., 9., 1., 7., 9. },{ 5., 6., 2., 8., 3., 1. } };
  MatD x = { { 7.}, {3.}, { 3. }, { 8. }, { 2. }, { 4. } };
  MatD y(1, A.cols(), 0.0);

  igm::blas::gemv(y, A, x);
  MatD r = { { 167.}, {91.}, {107.}, {133.} };
  for (size_t i = 0; i < r.size(); ++i)
  {
    ASSERT_EQ(r.at(i), y.at(i));
  }
}


TEST(gemv_blas, gemv_blas_2_special_case)
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

  for (size_t l = 0; l < R.rows()-1; ++l)
  {
    R.sub(l, l, l + 1, R._nc - 1);
    Q.subcols(l + 1);
    igm::blas::gemv(R, Q);

    R.subreset();
    Q.subreset();
  }

  Mat Res = { { 1., 0., 0., 0., 0. },
  { 167., 1., 0., 0., 0. },
  { 91., 108., 1., 0., 0. },
  { 107., 147., 113., 1., 0. },
  { 133., 164., 87., 91., 1. },
  };
  for (size_t i = 0; i < Res.size(); ++i)
  {
    ASSERT_EQ(Res.at(i), R.at(i));
  }

}

