#include <TNL/DiscreteMath.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

using namespace TNL;

void testPower( const size_t value, const size_t power, const size_t expectation ) {
   EXPECT_EQ( TNL::discretePow(value, power), expectation) << value<< " " << power;
}

TEST(TemplatesTestSuite, PowerTest) {
  testPower(0, 1, 0);
  testPower(1, 1, 1);

  testPower(0, 2, 0);
  testPower(1, 2, 1);
  testPower(2, 2, 4);

  testPower(0, 3, 0);
  testPower(1, 3, 1);
  testPower(2, 3, 8);
  testPower(3, 3, 27);

  testPower(0, 4, 0);
  testPower(1, 4, 1);
  testPower(2, 4, 16);
  testPower(3, 4, 81);
  testPower(4, 4, 256);
}

void testCombinations(const int k, const int n, const int expectation) {
   EXPECT_EQ(TNL::combinationsCount(k, n), expectation) << k << " " << n;
}

TEST(TemplatesTestSuite, CombinationsTest) {
  testCombinations(0, 1, 1);
  testCombinations(1, 1, 1);

  testCombinations(0, 2, 1);
  testCombinations(1, 2, 2);
  testCombinations(2, 2, 1);

  testCombinations(0, 3, 1);
  testCombinations(1, 3, 3);
  testCombinations(2, 3, 3);
  testCombinations(3, 3, 1);

  testCombinations(0, 4, 1);
  testCombinations(1, 4, 4);
  testCombinations(2, 4, 6);
  testCombinations(3, 4, 4);
  testCombinations(4, 4, 1);
}

void testCumulativeCombinations(const int k, const int n, const int expectation) {
   EXPECT_EQ(TNL::cumulativeCombinationsCount(k, n), expectation) << " k = " << k << " n =  " << n;
}

TEST(TemplatesTestSuite, cumulativeCombinationsTest) {
  testCumulativeCombinations(0, 1, 1);
  testCumulativeCombinations(1, 1, 2);

  testCumulativeCombinations(0, 2, 1);
  testCumulativeCombinations(1, 2, 3);
  testCumulativeCombinations(2, 2, 4);

  testCumulativeCombinations(0, 3, 1);
  testCumulativeCombinations(1, 3, 4);
  testCumulativeCombinations(2, 3, 7);
  testCumulativeCombinations(3, 3, 8);

  testCumulativeCombinations(0, 4, 1 );
  testCumulativeCombinations(1, 4, 5 );
  testCumulativeCombinations(2, 4, 11);
  testCumulativeCombinations(3, 4, 15);
  testCumulativeCombinations(4, 4, 16);
}

#endif

#include "main.h"
