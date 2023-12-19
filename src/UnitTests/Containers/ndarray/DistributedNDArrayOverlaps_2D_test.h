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
   const int overlapsX = get< 0 >( typename DistributedNDArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedNDArray::OverlapsType{} );

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
using DistributedNDArrayTypes =
   ::testing::Types< DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0 >,     // X, Y
                                                  std::index_sequence< 0, 1 >,  // permutation - should not matter
                                                  Devices::Host,
                                                  int,
                                                  std::index_sequence< 2, 3 > > >  // overlaps
#ifdef __CUDACC__
                     ,
                     DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0 >,     // X, Y
                                                  std::index_sequence< 0, 1 >,  // permutation - should not matter
                                                  Devices::Cuda,
                                                  int,
                                                  std::index_sequence< 2, 3 > > >  // overlaps
#endif
                     >;

TYPED_TEST_SUITE( DistributedNDArrayOverlaps_2D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArrayOverlaps_2D_test, checkSumOfLocalSizes )
{
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
   const int localSizeWithOverlaps = ( localSizeX + 2 * this->overlapsX ) * ( localSizeY + 2 * this->overlapsY );
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localSizeWithOverlaps );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forAll( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forAll( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

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
         if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
             || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY )
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
         if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
             || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY )
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

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

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
         if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
             || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY )
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
         if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
             || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY )
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

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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
test_helper_synchronize_D2Q5( DistributedArray& a,
                              int globalSize,
                              int rank,
                              const std::vector< BlockType >& decomposition,
                              const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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
test_helper_synchronize_D2Q9( DistributedArray& a,
                              int globalSize,
                              int rank,
                              const std::vector< BlockType >& decomposition,
                              const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );

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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
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

#include "../../main_mpi.h"
