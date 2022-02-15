#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template<typename Grid, bool isValid, int... dimensions>
void verifyDimensionSet(Grid& grid) {
   typename Grid::Coordinate coordinates(dimensions...);

   for (int i = 0; i < (int)sizeof...(dimensions); i++) {
      EXPECT_EQ(grid.getDimension(i), coordinates[i]) << "Verify, that index access is correct";
      EXPECT_EQ(grid.getDimensions(i), coordinates[i]) << "Verify, that index access is correct";

      auto repeatedDimensions = grid.getDimensions(i, i, i, i, i, i, i, i, i, i);

      EXPECT_EQ(repeatedDimensions.getSize(), 10) << "Verify, that all dimension indices are returned";

      for (int j = 0; j < repeatedDimensions.getSize(); j++)
         EXPECT_EQ(repeatedDimensions[j], coordinates[i]) << "Verify, that it is possible to request the same dimension multiple times";

      // TODO: Fix assertions in tests
      // EXPECT_ANY_THROW(grid.getDimension(-i - 1)) << "Verify, that throws on small dimension";
      // EXPECT_ANY_THROW(grid.getDimension((int)sizeof...(dimensions) + i)) << "Verify, that throws on large dimension";

      // EXPECT_ANY_THROW(grid.getDimensions(-i - 1)) << "Verify, that throws on negative dimension";
      // EXPECT_ANY_THROW(grid.getDimensions((int)sizeof...(dimensions) + i)) << "Verify, that throws on large dimension";
   }

   auto result = grid.getDimensions();

   EXPECT_EQ(coordinates, result) << "Verify, that dimension container accessor returns valid dimension";
}

template<typename Grid, bool isValid, int... dimensions>
void testIndexSet(Grid& grid) {
   std::ostringstream s;

   for (const auto& x: { dimensions... })
      s << x;

   if (isValid) {
      EXPECT_NO_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of" << s.str() << " doesn't cause assert";
   } else {
      EXPECT_ANY_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of " << s.str() << " causes assert";
      return;
   }

   verifyDimensionSet<Grid, isValid, dimensions...>(grid);
}

template<typename Grid, bool isValid, int... dimensions>
void testContainerSet(Grid& grid) {
   std::ostringstream s;

   for (const auto& x: { dimensions... })
      s << x;

   typename Grid::Coordinate coordinate(dimensions...);

   if (isValid) {
      EXPECT_NO_THROW(grid.setDimensions(coordinate)) << "Verify, that the set of" << s.str() << " doesn't cause assert";
   } else {
      EXPECT_ANY_THROW(grid.setDimensions(coordinate)) << "Verify, that the set of " << s.str() << " causes assert";
      return;
   }

   verifyDimensionSet<Grid, isValid, dimensions...>(grid);
}

#endif
