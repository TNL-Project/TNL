#include <TNL/Containers/BlockPartitioning.h>

#ifdef HAVE_GTEST
   #include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

// range [0, 9) split into 3 subintervals
TEST( BlockPartitioningTest, splitRange_3_subintervals )
{
   // result for subinterval no. 0
   const Subrange< int > result_0 = splitRange( 0, 9, 0, 3 );
   EXPECT_EQ( result_0.getBegin(), 0 );
   EXPECT_EQ( result_0.getEnd(), 3 );

   // result for subinterval no. 1
   const Subrange< int > result_1 = splitRange( 0, 9, 1, 3 );
   EXPECT_EQ( result_1.getBegin(), 3 );
   EXPECT_EQ( result_1.getEnd(), 6 );

   // result for subinterval no. 2
   const Subrange< int > result_2 = splitRange( 0, 9, 2, 3 );
   EXPECT_EQ( result_2.getBegin(), 6 );
   EXPECT_EQ( result_2.getEnd(), 9 );
}

// range [0, 10) split into 4 subintervals")
TEST( BlockPartitioningTest, splitRange_4_subintervals )
{
   // result for subinterval no. 0
   const Subrange< int > result_0 = splitRange( 0, 10, 0, 4 );
   EXPECT_EQ( result_0.getBegin(), 0 );
   EXPECT_EQ( result_0.getEnd(), 3 );

   // result for subinterval no. 1
   const Subrange< int > result_1 = splitRange( 0, 10, 1, 4 );
   EXPECT_EQ( result_1.getBegin(), 3 );
   EXPECT_EQ( result_1.getEnd(), 6 );

   // result for subinterval no. 2
   const Subrange< int > result_2 = splitRange( 0, 10, 2, 4 );
   EXPECT_EQ( result_2.getBegin(), 6 );
   EXPECT_EQ( result_2.getEnd(), 8 );

   // result for subinterval no. 2
   const Subrange< int > result_3 = splitRange( 0, 10, 3, 4 );
   EXPECT_EQ( result_3.getBegin(), 8 );
   EXPECT_EQ( result_3.getEnd(), 10 );
}

#endif

#include "../main.h"
