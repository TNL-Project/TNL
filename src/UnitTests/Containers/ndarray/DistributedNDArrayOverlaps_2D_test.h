#include <gtest/gtest.h>

#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArrayView.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::detail;

/*
 * Light check of DistributedNDArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArrayOverlaps_2D_test : public ::testing::Test
{
protected:
   using ValueType = typename DistributedNDArray::ValueType;
   using DeviceType = typename DistributedNDArray::DeviceType;
   using IndexType = typename DistributedNDArray::IndexType;
   using DistributedNDArrayType = DistributedNDArray;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedNDArrayType distributedNDArray;

   const int rank = TNL::MPI::GetRank( communicator );
   const int nproc = TNL::MPI::GetSize( communicator );

   // NOTE: decomposeBlockOptimal does not work for pure 2D blocks,
   //       working with 3D everywhere is actually simpler
   using BlockType = Block< 3, IndexType >;
   using MultiIndex = typename BlockType::CoordinatesType;

   BlockType globalBlock;
   std::vector< BlockType > decomposition;

   DistributedNDArrayOverlaps_2D_test()
   {
      globalBlock = { MultiIndex{ 0, 0, 0 }, MultiIndex{ globalSize, globalSize, 1 } };
      decomposition = decomposeBlockOptimal( globalBlock, nproc );
      const BlockType& localBlock = decomposition.at( rank );

      distributedNDArray.setSizes( globalSize, globalSize );
      distributedNDArray.template setDistribution< 0 >( localBlock.begin.x(), localBlock.end.x(), communicator );
      distributedNDArray.template setDistribution< 1 >( localBlock.begin.y(), localBlock.end.y(), communicator );
      distributedNDArray.allocate();
   }
};

// types for which DistributedNDArrayOverlaps_2D_test is instantiated
using DistributedNDArrayTypes = ::testing::Types<
   DistributedNDArray< NDArray<
      double,
      SizesHolder< int, 0, 0 >,     // X, Y
      std::index_sequence< 0, 1 >,  // permutation - should not matter
      Devices::Host,
      int,
      StaticSizesHolder< int, 2, 3 > > >,  // overlaps
   DistributedNDArray< NDArray<
      double,
      SizesHolder< int, 0, 0 >,     // X, Y
      std::index_sequence< 1, 0 >,  // permutation - X is the fastest dimension
      Devices::Host,
      int,
      StaticSizesHolder< int, 1, 1 > > >  // overlaps
#ifdef __CUDACC__
   ,
   DistributedNDArray< NDArray<
      double,
      SizesHolder< int, 0, 0 >,     // X, Y
      std::index_sequence< 0, 1 >,  // permutation - should not matter
      Devices::Cuda,
      int,
      StaticSizesHolder< int, 2, 3 > > >,  // overlaps
   DistributedNDArray< NDArray<
      double,
      SizesHolder< int, 0, 0 >,     // X, Y
      std::index_sequence< 1, 0 >,  // permutation - X is the fastest dimension
      Devices::Cuda,
      int,
      StaticSizesHolder< int, 1, 1 > > >  // overlaps
#endif
   >;

TYPED_TEST_SUITE( DistributedNDArrayOverlaps_2D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, checkSumOfLocalSizes )
{
   const int overlapX = this->distributedNDArray.template getOverlap< 0 >();
   const int overlapY = this->distributedNDArray.template getOverlap< 1 >();
   const auto localRangeX = this->distributedNDArray.template getLocalRange< 0 >();
   const auto localRangeY = this->distributedNDArray.template getLocalRange< 1 >();
   const int localSizeX = localRangeX.getEnd() - localRangeX.getBegin();
   const int localSizeY = localRangeY.getEnd() - localRangeY.getBegin();
   const int localSize = localSizeX * localSizeY;
   int sumOfLocalSizes = 0;
   TNL::MPI::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->communicator );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize * this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 0 >(), this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 1 >(), this->globalSize );
   const int localSizeWithOverlaps = ( localSizeX + 2 * overlapX ) * ( localSizeY + 2 * overlapY );
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localSizeWithOverlaps );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forAll( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forAll( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
             || gj >= localRangeY.getEnd() )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
      }
   }

   a.setValue( 0 );
   a.getView().forAll( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
             || gj >= localRangeY.getEnd() )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, forAll )
{
   test_helper_forAll( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forLocalInterior( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();
   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         if( gi < localRangeX.getBegin() + overlapX || gi >= localRangeX.getEnd() - overlapX
             || gj < localRangeY.getBegin() + overlapY || gj >= localRangeY.getEnd() - overlapY )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
      }
   }

   a.setValue( 0 );
   a.getView().forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         if( gi < localRangeX.getBegin() + overlapX || gi >= localRangeX.getEnd() - overlapX
             || gj < localRangeY.getBegin() + overlapY || gj >= localRangeY.getEnd() - overlapY )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, forLocalInterior )
{
   test_helper_forLocalInterior( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         if( gi < localRangeX.getBegin() + overlapX || gi >= localRangeX.getEnd() - overlapX
             || gj < localRangeY.getBegin() + overlapY || gj >= localRangeY.getEnd() - overlapY )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
      }
   }

   a.setValue( 0 );
   a.getView().forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         if( gi < localRangeX.getBegin() + overlapX || gi >= localRangeX.getEnd() - overlapX
             || gj < localRangeY.getBegin() + overlapY || gj >= localRangeY.getEnd() - overlapY )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, forLocalBoundary )
{
   test_helper_forLocalBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forGhosts( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
             || gj >= localRangeY.getEnd() )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
      }
   }

   a.setValue( 0 );
   a.getView().forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
             || gj >= localRangeY.getEnd() )
         {
            EXPECT_EQ( a.getElement( gi, gj ), 1 );
         }
         else {
            EXPECT_EQ( a.getElement( gi, gj ), 0 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, forGhosts )
{
   test_helper_forGhosts( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray, typename BlockType >
void
test_helper_synchronize_D2Q5(
   DistributedArray& a,
   int globalSize,
   int rank,
   const std::vector< BlockType >& decomposition,
   const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) = gi + gj;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.setSynchronizationPattern( NDArraySyncPatterns::D2Q5 );
   setNeighbors( s1, NDArraySyncPatterns::D2Q5, rank, decomposition, globalBlock );
   s1.synchronize( a );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         const auto value = a.getElement( gi, gj );
         // handle periodic boundaries
         const IndexType new_gi = ( gi + globalSize ) % globalSize;
         const IndexType new_gj = ( gj + globalSize ) % globalSize;
         // calculate the expected value
         IndexType expected_value = new_gi + new_gj;
         // corners are skipped in the D2Q5 synchronization pattern
         if( ( gi < localRangeX.getBegin() && gj < localRangeY.getBegin() )
             || ( gi < localRangeX.getBegin() && gj >= localRangeY.getEnd() )
             || ( gi >= localRangeX.getEnd() && gj < localRangeY.getBegin() )
             || ( gi >= localRangeX.getEnd() && gj >= localRangeY.getEnd() ) )
            expected_value = -1;
         // check the result
         EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj;
      }
   }

   a.setValue( -1 );
   a.getView().forAll( setter );
   DistributedNDArraySynchronizer< typename DistributedArray::ViewType > s2;
   s2.setSynchronizationPattern( NDArraySyncPatterns::D2Q5 );
   setNeighbors( s2, NDArraySyncPatterns::D2Q5, rank, decomposition, globalBlock );
   auto view = a.getView();
   s2.synchronize( view );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         const auto value = a.getElement( gi, gj );
         // handle periodic boundaries
         const IndexType new_gi = ( gi + globalSize ) % globalSize;
         const IndexType new_gj = ( gj + globalSize ) % globalSize;
         // calculate the expected value
         IndexType expected_value = new_gi + new_gj;
         // corners are skipped in the D2Q5 synchronization pattern
         if( ( gi < localRangeX.getBegin() && gj < localRangeY.getBegin() )
             || ( gi < localRangeX.getBegin() && gj >= localRangeY.getEnd() )
             || ( gi >= localRangeX.getEnd() && gj < localRangeY.getBegin() )
             || ( gi >= localRangeX.getEnd() && gj >= localRangeY.getEnd() ) )
            expected_value = -1;
         // check the result
         EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj;
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, synchronize_D2Q5 )
{
   test_helper_synchronize_D2Q5(
      this->distributedNDArray, this->globalSize, this->rank, this->decomposition, this->globalBlock );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray, typename BlockType >
void
test_helper_synchronize_D2Q9(
   DistributedArray& a,
   int globalSize,
   int rank,
   const std::vector< BlockType >& decomposition,
   const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) = gi + gj;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.setSynchronizationPattern( NDArraySyncPatterns::D2Q9 );
   setNeighbors( s1, NDArraySyncPatterns::D2Q9, rank, decomposition, globalBlock );
   s1.synchronize( a );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         const auto value = a.getElement( gi, gj );
         // handle periodic boundaries
         const IndexType new_gi = ( gi + globalSize ) % globalSize;
         const IndexType new_gj = ( gj + globalSize ) % globalSize;
         // calculate the expected value
         IndexType expected_value = new_gi + new_gj;
         // check the result
         EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj;
      }
   }

   a.setValue( -1 );
   a.getView().forAll( setter );
   DistributedNDArraySynchronizer< typename DistributedArray::ViewType > s2;
   s2.setSynchronizationPattern( NDArraySyncPatterns::D2Q9 );
   setNeighbors( s2, NDArraySyncPatterns::D2Q9, rank, decomposition, globalBlock );
   auto view = a.getView();
   s2.synchronize( view );

   for( IndexType gi = localRangeX.getBegin() - overlapX; gi < localRangeX.getEnd() + overlapX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapY; gj < localRangeY.getEnd() + overlapY; gj++ ) {
         const auto value = a.getElement( gi, gj );
         // handle periodic boundaries
         const IndexType new_gi = ( gi + globalSize ) % globalSize;
         const IndexType new_gj = ( gj + globalSize ) % globalSize;
         // calculate the expected value
         IndexType expected_value = new_gi + new_gj;
         // check the result
         EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj;
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, synchronize_D2Q9 )
{
   test_helper_synchronize_D2Q9(
      this->distributedNDArray, this->globalSize, this->rank, this->decomposition, this->globalBlock );
}

// Regression test for the receive-buffer staging (via setBufferOffsets):
// with shifted buffer offsets the receive regions of several directions overlap in the interior corner cells,
// so a staged copy-kernel unpack would be able to overwrite data placed into the array by a contiguous (unstaged) receive.
// The synchronizer must stage all colliding receives,
// which gives a deterministic application order where the corner (most specific) buffers are applied last.
// The tested types use the permutation <1,0> (fastest X dimension):
// the Left/Right columns are then non-contiguous (staged) and the Top/Bottom rows contiguous (direct) for unit overlaps,
// so the corner cells pit the staged Left/Right unpack against the direct Bottom/Top receive in the unpatched code.
template< typename DistributedArray, typename BlockType >
void
test_helper_synchronize_D2Q9_shifted(
   DistributedArray& a,
   int globalSize,
   int rank,
   const std::vector< BlockType >& decomposition,
   const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;
   using ValueType = typename DistributedArray::ValueType;

   const int overlapX = a.template getOverlap< 0 >();
   const int overlapY = a.template getOverlap< 1 >();

   // unit overlaps are required so that some receive regions are contiguous
   // (single rows or points) and both receive paths are exercised
   if( overlapX != 1 || overlapY != 1 )
      GTEST_SKIP() << "the test requires unit overlaps in both dimensions";

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const IndexType lx0 = localRangeX.getBegin();
   const IndexType ly0 = localRangeY.getBegin();
   const IndexType ex = localRangeX.getEnd();
   const IndexType ey = localRangeY.getEnd();
   const IndexType L0 = ex - lx0;
   const IndexType L1 = ey - ly0;
   auto a_view = a.getLocalView();

   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.setSynchronizationPattern( NDArraySyncPatterns::D2Q9 );
   setNeighbors( s1, NDArraySyncPatterns::D2Q9, rank, decomposition, globalBlock );

   // the first synchronization binds the array view and allocates the buffers
   // with the default (unshifted) offsets
   s1.synchronize( a );

   // shift the offsets so that the receive regions of different directions
   // overlap in the interior corner cells
   s1.setBufferOffsets( 1 );

   // fill the interior with a unique global function and the ghosts with the
   // same function plus a rank tag, so the writes of different neighbors can be
   // distinguished and the application order of colliding receives checked
   auto interior_setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - lx0, gj - ly0 ) = 1000 * gj + gi;
   };
   auto ghost_setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - lx0, gj - ly0 ) = 1000000 * ( rank + 1 ) + 1000 * gj + gi;
   };
   a.forAll( interior_setter );
   a.forGhosts( ghost_setter );

   s1.synchronize( a );

   // wraparound for periodic block boundaries
   auto wrap = [ = ]( IndexType gi ) -> IndexType
   {
      return ( ( gi % globalSize ) + globalSize ) % globalSize;
   };

   // value of the global function at the given coordinates
   auto global_value = [ & ]( IndexType gi, IndexType gj ) -> ValueType
   {
      return 1000 * wrap( gj ) + wrap( gi );
   };

   // tag identifying the data written by the given rank
   auto rank_tag = []( IndexType r ) -> ValueType
   {
      return 1000000 * ( r + 1 );
   };

   // rank of the block satisfying the given condition on its coordinates
   auto find_rank = [ & ]( auto condition ) -> IndexType
   {
      for( std::size_t i = 0; i < decomposition.size(); i++ )
         if( condition( decomposition[ i ] ) )
            return i;
      ADD_FAILURE() << "neighbor block not found in the decomposition";
      return -1;
   };

   // find the ranks of the corner neighbors
   const IndexType rBL = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.end.x() ) == wrap( lx0 ) && wrap( b.end.y() ) == wrap( ly0 );
      } );
   const IndexType rBR = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.begin.x() ) == wrap( ex ) && wrap( b.end.y() ) == wrap( ly0 );
      } );
   const IndexType rTL = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.end.x() ) == wrap( lx0 ) && wrap( b.begin.y() ) == wrap( ey );
      } );
   const IndexType rTR = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.begin.x() ) == wrap( ex ) && wrap( b.begin.y() ) == wrap( ey );
      } );
   // find the ranks of the edge neighbors (sharing the full edge of this block)
   const IndexType rL = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.end.x() ) == wrap( lx0 ) && wrap( b.begin.y() ) == wrap( ly0 ) && wrap( b.end.y() ) == wrap( ey );
      } );
   const IndexType rR = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.begin.x() ) == wrap( ex ) && wrap( b.begin.y() ) == wrap( ly0 ) && wrap( b.end.y() ) == wrap( ey );
      } );
   const IndexType rB = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.end.y() ) == wrap( ly0 ) && wrap( b.begin.x() ) == wrap( lx0 ) && wrap( b.end.x() ) == wrap( ex );
      } );
   const IndexType rT = find_rank(
      [ & ]( const BlockType& b )
      {
         return wrap( b.begin.y() ) == wrap( ey ) && wrap( b.begin.x() ) == wrap( lx0 ) && wrap( b.end.x() ) == wrap( ex );
      } );

   if( rBL < 0 || rBR < 0 || rTL < 0 || rTR < 0 || rL < 0 || rR < 0 || rB < 0 || rT < 0 )
      return;

   // value with which the ghost cell at the given global coordinates was
   // filled on the given rank (the coordinates may lie outside the global
   // block, so no wraparound is applied here)
   auto ghost_value = [ & ]( IndexType gi, IndexType gj, IndexType r ) -> ValueType
   {
      return rank_tag( r ) + 1000 * gj + gi;
   };

   const BlockType& BLK = decomposition[ rBL ];
   const BlockType& BRK = decomposition[ rBR ];
   const BlockType& TLK = decomposition[ rTL ];
   const BlockType& TRK = decomposition[ rTR ];
   const BlockType& LNK = decomposition[ rL ];
   const BlockType& RNK = decomposition[ rR ];
   const BlockType& BNK = decomposition[ rB ];
   const BlockType& TNK = decomposition[ rT ];

   auto value_at = [ & ]( IndexType x, IndexType y ) -> ValueType
   {
      return a.getElement( lx0 + x, ly0 + y );
   };

   if( L0 > 2 && L1 > 2 ) {
      const IndexType xm = L0 / 2;
      const IndexType ym = L1 / 2;
      // cells that are covered only by a single (edge) buffer
      EXPECT_EQ( value_at( 0, ym ), ghost_value( LNK.end.x(), LNK.begin.y() + ym, rL ) );
      EXPECT_EQ( value_at( L0 - 1, ym ), ghost_value( RNK.begin.x() - 1, RNK.begin.y() + ym, rR ) );
      EXPECT_EQ( value_at( xm, 0 ), ghost_value( BNK.begin.x() + xm, BNK.end.y(), rB ) );
      EXPECT_EQ( value_at( xm, L1 - 1 ), ghost_value( TNK.begin.x() + xm, TNK.begin.y() - 1, rT ) );
      // cells that are covered by no receive buffer are not modified
      EXPECT_EQ( value_at( xm, ym ), global_value( lx0 + xm, ly0 + ym ) );
   }
   // corner cells are contested by several buffers: the corner buffer must be
   // applied last, so the data must come from the corner neighbor
   EXPECT_EQ( value_at( 0, 0 ), ghost_value( BLK.end.x(), BLK.end.y(), rBL ) );
   EXPECT_EQ( value_at( L0 - 1, 0 ), ghost_value( BRK.begin.x() - 1, BRK.end.y(), rBR ) );
   EXPECT_EQ( value_at( 0, L1 - 1 ), ghost_value( TLK.end.x(), TLK.begin.y() - 1, rTL ) );
   EXPECT_EQ( value_at( L0 - 1, L1 - 1 ), ghost_value( TRK.begin.x() - 1, TRK.begin.y() - 1, rTR ) );
}

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, synchronize_D2Q9_shifted_offsets )
{
   test_helper_synchronize_D2Q9_shifted(
      this->distributedNDArray, this->globalSize, this->rank, this->decomposition, this->globalBlock );
}

#include "../../main_mpi.h"
