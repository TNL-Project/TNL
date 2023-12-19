#include <TNL/Containers/BlockPartitioning.h>

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

// decomposition along x-axis
TEST( BlockPartitioningTest, decomposeBlock_along_x )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 3, 1, 1 );
   ASSERT_EQ( result.size(), 3 );

   EXPECT_EQ( result[ 0 ].begin, global.begin );
   EXPECT_EQ( result[ 0 ].end, idx3d( 40, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( 40, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( 70, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( 70, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, global.end );
}

// decomposition along y-axis
TEST( BlockPartitioningTest, decomposeBlock_along_y )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 1, 3, 1 );
   ASSERT_EQ( result.size(), 3 );

   EXPECT_EQ( result[ 0 ].begin, global.begin );
   EXPECT_EQ( result[ 0 ].end, idx3d( global.end.x(), 80, global.end.z() ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( global.begin.x(), 80, global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( global.end.x(), 140, global.end.z() ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( global.begin.x(), 140, global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, global.end );
}

// decomposition along z-axis
TEST( BlockPartitioningTest, decomposeBlock_along_z )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 1, 1, 3 );
   ASSERT_EQ( result.size(), 3 );

   EXPECT_EQ( result[ 0 ].begin, global.begin );
   EXPECT_EQ( result[ 0 ].end, idx3d( global.end.x(), global.end.y(), 120 ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( global.begin.x(), global.begin.y(), 120 ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( global.end.x(), global.end.y(), 210 ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( global.begin.x(), global.begin.y(), 210 ) );
   EXPECT_EQ( result[ 2 ].end, global.end );
}

// decomposition along x- and y- axes
TEST( BlockPartitioningTest, decomposeBlock_along_xy )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 3, 2, 1 );
   ASSERT_EQ( result.size(), 6 );

   EXPECT_EQ( result[ 0 ].begin, idx3d( global.begin.x(), global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 0 ].end, idx3d( 40, 110, global.end.z() ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( 40, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( 70, 110, global.end.z() ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( 70, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, idx3d( global.end.x(), 110, global.end.z() ) );

   EXPECT_EQ( result[ 3 ].begin, idx3d( global.begin.x(), 110, global.begin.z() ) );
   EXPECT_EQ( result[ 3 ].end, idx3d( 40, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 4 ].begin, idx3d( 40, 110, global.begin.z() ) );
   EXPECT_EQ( result[ 4 ].end, idx3d( 70, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 5 ].begin, idx3d( 70, 110, global.begin.z() ) );
   EXPECT_EQ( result[ 5 ].end, idx3d( global.end.x(), global.end.y(), global.end.z() ) );
}

// decomposition along x- and z- axes
TEST( BlockPartitioningTest, decomposeBlock_along_xz )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 3, 1, 2 );
   ASSERT_EQ( result.size(), 6 );

   EXPECT_EQ( result[ 0 ].begin, idx3d( global.begin.x(), global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 0 ].end, idx3d( 40, global.end.y(), 165 ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( 40, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( 70, global.end.y(), 165 ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( 70, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, idx3d( global.end.x(), global.end.y(), 165 ) );

   EXPECT_EQ( result[ 3 ].begin, idx3d( global.begin.x(), global.begin.y(), 165 ) );
   EXPECT_EQ( result[ 3 ].end, idx3d( 40, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 4 ].begin, idx3d( 40, global.begin.y(), 165 ) );
   EXPECT_EQ( result[ 4 ].end, idx3d( 70, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 5 ].begin, idx3d( 70, global.begin.y(), 165 ) );
   EXPECT_EQ( result[ 5 ].end, idx3d( global.end.x(), global.end.y(), global.end.z() ) );
}

// decomposition along y- and z- axes
TEST( BlockPartitioningTest, decomposeBlock_along_yz )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 1, 3, 2 );
   ASSERT_EQ( result.size(), 6 );

   EXPECT_EQ( result[ 0 ].begin, idx3d( global.begin.x(), global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 0 ].end, idx3d( global.end.x(), 80, 165 ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( global.begin.x(), 80, global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( global.end.x(), 140, 165 ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( global.begin.x(), 140, global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, idx3d( global.end.x(), global.end.y(), 165 ) );

   EXPECT_EQ( result[ 3 ].begin, idx3d( global.begin.x(), global.begin.y(), 165 ) );
   EXPECT_EQ( result[ 3 ].end, idx3d( global.end.x(), 80, global.end.z() ) );

   EXPECT_EQ( result[ 4 ].begin, idx3d( global.begin.x(), 80, 165 ) );
   EXPECT_EQ( result[ 4 ].end, idx3d( global.end.x(), 140, global.end.z() ) );

   EXPECT_EQ( result[ 5 ].begin, idx3d( global.begin.x(), 140, 165 ) );
   EXPECT_EQ( result[ 5 ].end, idx3d( global.end.x(), global.end.y(), global.end.z() ) );
}

// decomposition along all three axes
TEST( BlockPartitioningTest, decomposeBlock_along_xyz )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 10, 20, 30 );
   global.end = idx3d( 100, 200, 300 );

   const std::vector< block3d > result = decomposeBlock( global, 3, 3, 3 );
   ASSERT_EQ( result.size(), 27 );

   EXPECT_EQ( result[ 0 ].begin, idx3d( global.begin.x(), global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 0 ].end, idx3d( 40, 80, 120 ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( 40, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( 70, 80, 120 ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( 70, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, idx3d( global.end.x(), 80, 120 ) );

   EXPECT_EQ( result[ 3 ].begin, idx3d( global.begin.x(), 80, global.begin.z() ) );
   EXPECT_EQ( result[ 3 ].end, idx3d( 40, 140, 120 ) );

   EXPECT_EQ( result[ 4 ].begin, idx3d( 40, 80, global.begin.z() ) );
   EXPECT_EQ( result[ 4 ].end, idx3d( 70, 140, 120 ) );

   EXPECT_EQ( result[ 5 ].begin, idx3d( 70, 80, global.begin.z() ) );
   EXPECT_EQ( result[ 5 ].end, idx3d( global.end.x(), 140, 120 ) );

   EXPECT_EQ( result[ 6 ].begin, idx3d( global.begin.x(), 140, global.begin.z() ) );
   EXPECT_EQ( result[ 6 ].end, idx3d( 40, global.end.y(), 120 ) );

   EXPECT_EQ( result[ 7 ].begin, idx3d( 40, 140, global.begin.z() ) );
   EXPECT_EQ( result[ 7 ].end, idx3d( 70, global.end.y(), 120 ) );

   EXPECT_EQ( result[ 8 ].begin, idx3d( 70, 140, global.begin.z() ) );
   EXPECT_EQ( result[ 8 ].end, idx3d( global.end.x(), global.end.y(), 120 ) );

   EXPECT_EQ( result[ 9 ].begin, idx3d( global.begin.x(), global.begin.y(), 120 ) );
   EXPECT_EQ( result[ 9 ].end, idx3d( 40, 80, 210 ) );

   EXPECT_EQ( result[ 10 ].begin, idx3d( 40, global.begin.y(), 120 ) );
   EXPECT_EQ( result[ 10 ].end, idx3d( 70, 80, 210 ) );

   EXPECT_EQ( result[ 11 ].begin, idx3d( 70, global.begin.y(), 120 ) );
   EXPECT_EQ( result[ 11 ].end, idx3d( global.end.x(), 80, 210 ) );

   EXPECT_EQ( result[ 12 ].begin, idx3d( global.begin.x(), 80, 120 ) );
   EXPECT_EQ( result[ 12 ].end, idx3d( 40, 140, 210 ) );

   EXPECT_EQ( result[ 13 ].begin, idx3d( 40, 80, 120 ) );
   EXPECT_EQ( result[ 13 ].end, idx3d( 70, 140, 210 ) );

   EXPECT_EQ( result[ 14 ].begin, idx3d( 70, 80, 120 ) );
   EXPECT_EQ( result[ 14 ].end, idx3d( global.end.x(), 140, 210 ) );

   EXPECT_EQ( result[ 15 ].begin, idx3d( global.begin.x(), 140, 120 ) );
   EXPECT_EQ( result[ 15 ].end, idx3d( 40, global.end.y(), 210 ) );

   EXPECT_EQ( result[ 16 ].begin, idx3d( 40, 140, 120 ) );
   EXPECT_EQ( result[ 16 ].end, idx3d( 70, global.end.y(), 210 ) );

   EXPECT_EQ( result[ 17 ].begin, idx3d( 70, 140, 120 ) );
   EXPECT_EQ( result[ 17 ].end, idx3d( global.end.x(), global.end.y(), 210 ) );

   EXPECT_EQ( result[ 18 ].begin, idx3d( global.begin.x(), global.begin.y(), 210 ) );
   EXPECT_EQ( result[ 18 ].end, idx3d( 40, 80, global.end.z() ) );

   EXPECT_EQ( result[ 19 ].begin, idx3d( 40, global.begin.y(), 210 ) );
   EXPECT_EQ( result[ 19 ].end, idx3d( 70, 80, global.end.z() ) );

   EXPECT_EQ( result[ 20 ].begin, idx3d( 70, global.begin.y(), 210 ) );
   EXPECT_EQ( result[ 20 ].end, idx3d( global.end.x(), 80, global.end.z() ) );

   EXPECT_EQ( result[ 21 ].begin, idx3d( global.begin.x(), 80, 210 ) );
   EXPECT_EQ( result[ 21 ].end, idx3d( 40, 140, global.end.z() ) );

   EXPECT_EQ( result[ 22 ].begin, idx3d( 40, 80, 210 ) );
   EXPECT_EQ( result[ 22 ].end, idx3d( 70, 140, global.end.z() ) );

   EXPECT_EQ( result[ 23 ].begin, idx3d( 70, 80, 210 ) );
   EXPECT_EQ( result[ 23 ].end, idx3d( global.end.x(), 140, global.end.z() ) );

   EXPECT_EQ( result[ 24 ].begin, idx3d( global.begin.x(), 140, 210 ) );
   EXPECT_EQ( result[ 24 ].end, idx3d( 40, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 25 ].begin, idx3d( 40, 140, 210 ) );
   EXPECT_EQ( result[ 25 ].end, idx3d( 70, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 26 ].begin, idx3d( 70, 140, 210 ) );
   EXPECT_EQ( result[ 26 ].end, idx3d( global.end.x(), global.end.y(), global.end.z() ) );
}

// optimal decomposition along x-axis
TEST( BlockPartitioningTest, decomposeBlockOptimal_along_x )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // create "global" lattice to be decomposed
   block3d global;
   global.begin = idx3d( 0, 0, 0 );
   global.end = idx3d( 300, 100, 100 );

   const std::vector< block3d > result = decomposeBlockOptimal( global, 3 );
   ASSERT_EQ( result.size(), 3 );

   EXPECT_EQ( result[ 0 ].begin, global.begin );
   EXPECT_EQ( result[ 0 ].end, idx3d( 100, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 1 ].begin, idx3d( 100, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 1 ].end, idx3d( 200, global.end.y(), global.end.z() ) );

   EXPECT_EQ( result[ 2 ].begin, idx3d( 200, global.begin.y(), global.begin.z() ) );
   EXPECT_EQ( result[ 2 ].end, global.end );
}

#include "../main.h"
