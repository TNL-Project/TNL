#include <TNL/Containers/Block.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

TEST( BlockTest, Block3D )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   static_assert( std::is_same_v< typename block3d::idx, int >, "unexpected idx type" );
   static_assert( std::is_same_v< typename block3d::CoordinatesType, idx3d >, "unexpected CoordinatesType" );

   // default constructor
   {
      const idx3d zero = 0;
      block3d block;
      EXPECT_EQ( block.dimension, 3 );
      EXPECT_EQ( block.begin, zero );
      EXPECT_EQ( block.end, zero );
   }

   // constructor with begin and end
   {
      const idx3d begin = 0;
      const idx3d end = 1;
      block3d block( begin, end );
      EXPECT_EQ( block.dimension, 3 );
      EXPECT_EQ( block.begin, begin );
      EXPECT_EQ( block.end, end );
   }

   // lexicographic ordering
   {
      block3d block_1( { 0, 1, 2 }, { 3, 4, 5 } );
      block3d block_2( { 1, 2, 2 }, { 3, 4, 5 } );
      block3d block_3( { 2, 3, 4 }, { 3, 4, 5 } );
      EXPECT_LT( block_1, block_2 );
      EXPECT_LT( block_1, block_3 );
      EXPECT_LT( block_2, block_3 );
      EXPECT_FALSE( block_1 < block_1 );
      EXPECT_FALSE( block_2 < block_1 );
      EXPECT_FALSE( block_3 < block_1 );
      EXPECT_FALSE( block_3 < block_2 );
   }
}

TEST( BlockPartitioningTest, getVolume_1D_block )
{
   using idx1d = StaticVector< 1, int >;
   using block1d = Block< 1, int >;

   const idx1d begin = 1;
   const idx1d end = 3;
   const block1d block( begin, end );
   EXPECT_EQ( getVolume( block ), end.x() - begin.x() );
}

TEST( BlockPartitioningTest, getVolume_2D_block )
{
   using idx2d = StaticVector< 2, int >;
   using block2d = Block< 2, int >;

   const idx2d begin = 1;
   const idx2d end = 3;
   const block2d block( begin, end );
   EXPECT_EQ( getVolume( block ), 2 * 2 );
}

TEST( BlockPartitioningTest, getVolume_3D_block )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   const idx3d begin = 1;
   const idx3d end = 3;
   const block3d block( begin, end );
   EXPECT_EQ( getVolume( block ), 2 * 2 * 2 );
}

TEST( BlockPartitioningTest, getVolume_decomposition )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // this is not really a decomposition (blocks are adjacent across a vertex),
   // but still allows to test the volume function
   const idx3d p1 = 1;
   const idx3d p2 = 2;
   const idx3d p3 = 3;
   const idx3d p4 = 4;
   const std::vector< block3d > blocks = { { p1, p2 }, { p2, p3 }, { p3, p4 } };
   EXPECT_EQ( getVolume( blocks ), 3 );
}

TEST( BlockPartitioningTest, getMaximumImbalance )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   const idx3d b1_begin = { 0, 0, 0 };
   const idx3d b1_end = { 1, 1, 1 };
   const idx3d b2_begin = { 1, 0, 0 };
   const idx3d b2_end = { 2, 1, 1 };
   const idx3d b3_begin = { 2, 0, 0 };
   const idx3d b3_end = { 3, 1, 1 };
   const idx3d b4_begin = { 3, 0, 0 };
   const idx3d b4_end = { 4, 1, 1 };

   // balanced decomposition
   {
      const std::vector< block3d > decomposition = {
         { b1_begin, b1_end }, { b2_begin, b2_end }, { b3_begin, b3_end }, { b4_begin, b4_end }
      };
      const int global_volume = getVolume( decomposition );
      ASSERT_EQ( global_volume, 4 );
      EXPECT_EQ( getMaximumImbalance( decomposition ), 0 );
      EXPECT_EQ( getMaximumImbalance( decomposition, global_volume ), 0 );
   }

   // imbalanced decomposition
   {
      const std::vector< block3d > decomposition = { { b1_begin, b1_end }, { b2_begin, b4_end } };
      const int global_volume = getVolume( decomposition );
      ASSERT_EQ( global_volume, 4 );
      EXPECT_EQ( getMaximumImbalance( decomposition ), 0.5 );
      EXPECT_EQ( getMaximumImbalance( decomposition, global_volume ), 0.5 );
   }
}

TEST( BlockPartitioningTest, getArea )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // x-normal
   {
      const idx3d begin = { 0, 1, 2 };
      const idx3d end = { 0, 3, 4 };
      const block3d block( begin, end );
      EXPECT_EQ( getArea( block ), 2 * 2 );
   }

   // y-normal
   {
      const idx3d begin = { 1, 0, 2 };
      const idx3d end = { 3, 0, 4 };
      const block3d block( begin, end );
      EXPECT_EQ( getArea( block ), 2 * 2 );
   }

   // z-normal
   {
      const idx3d begin = { 1, 2, 0 };
      const idx3d end = { 3, 4, 0 };
      const block3d block( begin, end );
      EXPECT_EQ( getArea( block ), 2 * 2 );
   }
}

TEST( BlockPartitioningTest, createSides_in_3D )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // one cube to vector
   {
      const idx3d begin = { 0, 0, 0 };
      const idx3d end = { 1, 2, 3 };
      const block3d block( begin, end );

      std::vector< block3d > sides;
      createSides( block, std::inserter( sides, sides.end() ) );
      ASSERT_EQ( sides.size(), 6 );
      EXPECT_EQ( sides[ 0 ], ( block3d{ { 0, 0, 0 }, { 1, 2, 0 } } ) );
      EXPECT_EQ( sides[ 1 ], ( block3d{ { 0, 0, 0 }, { 1, 0, 3 } } ) );
      EXPECT_EQ( sides[ 2 ], ( block3d{ { 0, 0, 0 }, { 0, 2, 3 } } ) );
      EXPECT_EQ( sides[ 3 ], ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 4 ], ( block3d{ { 0, 2, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 5 ], ( block3d{ { 0, 0, 3 }, { 1, 2, 3 } } ) );
   }

   // one cube to set
   {
      const idx3d begin = { 0, 0, 0 };
      const idx3d end = { 1, 2, 3 };
      const std::vector< block3d > blocks{ { begin, end } };

      std::set< block3d > sides_set = createSides( blocks );
      ASSERT_EQ( sides_set.size(), 6 );

      std::vector< block3d > sides( sides_set.begin(), sides_set.end() );
      EXPECT_EQ( sides[ 0 ], ( block3d{ { 0, 0, 0 }, { 0, 2, 3 } } ) );
      EXPECT_EQ( sides[ 1 ], ( block3d{ { 0, 0, 0 }, { 1, 0, 3 } } ) );
      EXPECT_EQ( sides[ 2 ], ( block3d{ { 0, 0, 0 }, { 1, 2, 0 } } ) );
      EXPECT_EQ( sides[ 3 ], ( block3d{ { 0, 0, 3 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 4 ], ( block3d{ { 0, 2, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 5 ], ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );
   }

   // two cubes to vector
   {
      const idx3d begin_1 = { 0, 0, 0 };
      const idx3d end_1 = { 1, 2, 3 };
      const idx3d begin_2 = { 1, 0, 0 };
      const idx3d end_2 = { 2, 2, 3 };
      const std::vector< block3d > blocks{ { begin_1, end_1 }, { begin_2, end_2 } };

      std::vector< block3d > sides;
      createSides( blocks[ 0 ], std::inserter( sides, sides.end() ) );
      createSides( blocks[ 1 ], std::inserter( sides, sides.end() ) );
      ASSERT_EQ( sides.size(), 12 );

      EXPECT_EQ( sides[ 0 ], ( block3d{ { 0, 0, 0 }, { 1, 2, 0 } } ) );
      EXPECT_EQ( sides[ 1 ], ( block3d{ { 0, 0, 0 }, { 1, 0, 3 } } ) );
      EXPECT_EQ( sides[ 2 ], ( block3d{ { 0, 0, 0 }, { 0, 2, 3 } } ) );
      EXPECT_EQ( sides[ 3 ], ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 4 ], ( block3d{ { 0, 2, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 5 ], ( block3d{ { 0, 0, 3 }, { 1, 2, 3 } } ) );

      EXPECT_EQ( sides[ 6 ], ( block3d{ { 1, 0, 0 }, { 2, 2, 0 } } ) );
      EXPECT_EQ( sides[ 7 ], ( block3d{ { 1, 0, 0 }, { 2, 0, 3 } } ) );
      EXPECT_EQ( sides[ 8 ], ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 9 ], ( block3d{ { 2, 0, 0 }, { 2, 2, 3 } } ) );
      EXPECT_EQ( sides[ 10 ], ( block3d{ { 1, 2, 0 }, { 2, 2, 3 } } ) );
      EXPECT_EQ( sides[ 11 ], ( block3d{ { 1, 0, 3 }, { 2, 2, 3 } } ) );
   }

   // two cubes to set
   {
      const idx3d begin_1 = { 0, 0, 0 };
      const idx3d end_1 = { 1, 2, 3 };
      const idx3d begin_2 = { 1, 0, 0 };
      const idx3d end_2 = { 2, 2, 3 };
      const std::vector< block3d > blocks{ { begin_1, end_1 }, { begin_2, end_2 } };

      std::set< block3d > sides_set = createSides( blocks );
      ASSERT_EQ( sides_set.size(), 11 );

      std::vector< block3d > sides( sides_set.begin(), sides_set.end() );

      EXPECT_EQ( sides[ 0 ], ( block3d{ { 0, 0, 0 }, { 0, 2, 3 } } ) );
      EXPECT_EQ( sides[ 1 ], ( block3d{ { 0, 0, 0 }, { 1, 0, 3 } } ) );
      EXPECT_EQ( sides[ 2 ], ( block3d{ { 0, 0, 0 }, { 1, 2, 0 } } ) );
      EXPECT_EQ( sides[ 3 ], ( block3d{ { 0, 0, 3 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 4 ], ( block3d{ { 0, 2, 0 }, { 1, 2, 3 } } ) );
      EXPECT_EQ( sides[ 5 ], ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );

      EXPECT_EQ( sides[ 6 ], ( block3d{ { 1, 0, 0 }, { 2, 0, 3 } } ) );
      EXPECT_EQ( sides[ 7 ], ( block3d{ { 1, 0, 0 }, { 2, 2, 0 } } ) );
      EXPECT_EQ( sides[ 8 ], ( block3d{ { 1, 0, 3 }, { 2, 2, 3 } } ) );
      EXPECT_EQ( sides[ 9 ], ( block3d{ { 1, 2, 0 }, { 2, 2, 3 } } ) );
      EXPECT_EQ( sides[ 10 ], ( block3d{ { 2, 0, 0 }, { 2, 2, 3 } } ) );
   }
}

TEST( BlockPartitioningTest, createInteriorSides_in_3D )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // two cubes
   {
      const idx3d begin_1 = { 0, 0, 0 };
      const idx3d end_1 = { 1, 2, 3 };
      const idx3d begin_2 = { 1, 0, 0 };
      const idx3d end_2 = { 2, 2, 3 };
      const std::vector< block3d > blocks{ { begin_1, end_1 }, { begin_2, end_2 } };

      std::set< block3d > sides = createInteriorSides( blocks );
      ASSERT_EQ( sides.size(), 1 );
      EXPECT_EQ( *sides.begin(), ( block3d{ { 1, 0, 0 }, { 1, 2, 3 } } ) );
   }
}

TEST( BlockPartitioningTest, getInterfaceArea_in_3D )
{
   using idx3d = StaticVector< 3, int >;
   using block3d = Block< 3, int >;

   // two cubes
   {
      const idx3d begin_1 = { 0, 0, 0 };
      const idx3d end_1 = { 1, 2, 3 };
      const idx3d begin_2 = { 1, 0, 0 };
      const idx3d end_2 = { 2, 2, 3 };
      const std::vector< block3d > blocks{ { begin_1, end_1 }, { begin_2, end_2 } };

      EXPECT_EQ( getInterfaceArea( blocks ), 6 );
   }
}

#include "../main.h"
