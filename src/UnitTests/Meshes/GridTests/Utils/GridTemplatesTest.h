#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include <TNL/Meshes/GridDetails/Templates/Templates.h>

void testCombination(const int k, const int n, const int expectation) {
   EXPECT_EQ(TNL::Meshes::Templates::combination(k, n), expectation) << k << " " << n;
}

TEST(TemplatesTestSuite, CombinationsTest) {
  testCombination(0, 1, 1);
  testCombination(1, 1, 1);

  testCombination(0, 2, 1);
  testCombination(1, 2, 2);
  testCombination(2, 2, 1);

  testCombination(0, 3, 1);
  testCombination(1, 3, 3);
  testCombination(2, 3, 3);
  testCombination(3, 3, 1);

  testCombination(0, 4, 1);
  testCombination(1, 4, 4);
  testCombination(2, 4, 6);
  testCombination(3, 4, 4);
  testCombination(4, 4, 1);
}

#endif
