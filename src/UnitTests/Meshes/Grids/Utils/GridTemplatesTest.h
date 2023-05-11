#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/GridDetails/Templates/Functions.h>
#include "../CoordinateIterator.h"

template<int Size>
using Coordinate = TNL::Containers::StaticVector<Size, int>;

template<int Size>
void testIndexCollapse(const int base) {
   SCOPED_TRACE("Coordinate size: " + TNL::convertToString(Size));
   SCOPED_TRACE("Base: " + TNL::convertToString(base));

   const int halfBase = base >> 1;
   Coordinate<Size> start, end;

   for (int i = 0; i < Size; i++) {
      start[i] = -halfBase;
      // Want to traverse
      end[i] = halfBase + 1;
   }

   CoordinateIterator<int, Size> iterator(start, end);

   int index = 0;

   do {
      EXPECT_EQ(TNL::Meshes::Templates::makeCollapsedIndex(base, iterator.getCoordinate()), index)
         << base << " " << index << " " << iterator.getCoordinate();
      index++;
   } while (!iterator.next());
}

TEST(TemplatesTestSuite, IndexCollapseTest) {
   testIndexCollapse<1>(3);
   testIndexCollapse<2>(3);
   testIndexCollapse<3>(3);
   testIndexCollapse<4>(3);

   testIndexCollapse<1>(5);
   testIndexCollapse<2>(5);
   testIndexCollapse<3>(5);
   testIndexCollapse<4>(5);
}

#endif
