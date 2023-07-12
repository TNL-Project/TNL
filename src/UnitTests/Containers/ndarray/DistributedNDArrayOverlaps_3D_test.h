#ifdef HAVE_GTEST
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
 * - Global size is hardcoded as 31 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArrayOverlaps_3D_test : public ::testing::Test
{
protected:
   using ValueType = typename DistributedNDArray::ValueType;
   using DeviceType = typename DistributedNDArray::DeviceType;
   using IndexType = typename DistributedNDArray::IndexType;
   using DistributedNDArrayType = DistributedNDArray;

   const int globalSize = 31;  // prime number to force non-uniform distribution
   const int overlapsX = get< 0 >( typename DistributedNDArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedNDArray::OverlapsType{} );
   const int overlapsZ = get< 2 >( typename DistributedNDArray::OverlapsType{} );

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedNDArrayType distributedNDArray;

   const int rank = TNL::MPI::GetRank( communicator );
   const int nproc = TNL::MPI::GetSize( communicator );

   using BlockType = Block< 3, IndexType >;
   using MultiIndex = typename BlockType::CoordinatesType;

   BlockType globalBlock;
   std::vector< BlockType > decomposition;

   DistributedNDArrayOverlaps_3D_test()
   {
      globalBlock = { MultiIndex{ 0, 0, 0 }, MultiIndex{ globalSize, globalSize, globalSize } };
      decomposition = decomposeBlockOptimal( globalBlock, nproc );
      const BlockType& localBlock = decomposition.at( rank );

      distributedNDArray.setSizes( globalSize, globalSize, globalSize );
      distributedNDArray.template setDistribution< 0 >( localBlock.begin.x(), localBlock.end.x(), communicator );
      distributedNDArray.template setDistribution< 1 >( localBlock.begin.y(), localBlock.end.y(), communicator );
      distributedNDArray.template setDistribution< 2 >( localBlock.begin.z(), localBlock.end.z(), communicator );
      distributedNDArray.allocate();
   }
};

// types for which DistributedNDArrayOverlaps_3D_test is instantiated
using DistributedNDArrayTypes =
   ::testing::Types< DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0, 0 >,     // X, Y, Z
                                                  std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                                  Devices::Host,
                                                  int,
                                                  std::index_sequence< 2, 3, 1 > > >  // overlaps
   #ifdef __CUDACC__
                     ,
                     DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0, 0 >,     // X, Y, Z
                                                  std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                                  Devices::Cuda,
                                                  int,
                                                  std::index_sequence< 2, 3, 1 > > >  // overlaps
   #endif
                     >;

TYPED_TEST_SUITE( DistributedNDArrayOverlaps_3D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, checkSumOfLocalSizes )
{
   const auto localRangeX = this->distributedNDArray.template getLocalRange< 0 >();
   const auto localRangeY = this->distributedNDArray.template getLocalRange< 1 >();
   const auto localRangeZ = this->distributedNDArray.template getLocalRange< 2 >();
   const int localSizeX = localRangeX.getEnd() - localRangeX.getBegin();
   const int localSizeY = localRangeY.getEnd() - localRangeY.getBegin();
   const int localSizeZ = localRangeZ.getEnd() - localRangeZ.getBegin();
   const int localSize = localSizeX * localSizeY * localSizeZ;
   int sumOfLocalSizes = 0;
   TNL::MPI::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->communicator );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize * this->globalSize * this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 0 >(), this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 1 >(), this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 2 >(), this->globalSize );
   const int localSizeWithOverlaps =
      ( localSizeX + 2 * this->overlapsX ) * ( localSizeY + 2 * this->overlapsY ) * ( localSizeZ + 2 * this->overlapsZ );
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
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forAll( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
                || gj >= localRangeY.getEnd() || gk < localRangeZ.getBegin() || gk >= localRangeZ.getEnd() )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
         }
      }
   }

   a.setValue( 0 );
   a.getView().forAll( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
                || gj >= localRangeY.getEnd() || gk < localRangeZ.getBegin() || gk >= localRangeZ.getEnd() )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, forAll )
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
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();
   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
                || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY
                || gk < localRangeZ.getBegin() + overlapsZ || gk >= localRangeZ.getEnd() - overlapsZ )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
         }
      }
   }

   a.setValue( 0 );
   a.getView().forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
                || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY
                || gk < localRangeZ.getBegin() + overlapsZ || gk >= localRangeZ.getEnd() - overlapsZ )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, forLocalInterior )
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
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
                || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY
                || gk < localRangeZ.getBegin() + overlapsZ || gk >= localRangeZ.getEnd() - overlapsZ )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
         }
      }
   }

   a.setValue( 0 );
   a.getView().forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi < localRangeX.getBegin() + overlapsX || gi >= localRangeX.getEnd() - overlapsX
                || gj < localRangeY.getBegin() + overlapsY || gj >= localRangeY.getEnd() - overlapsY
                || gk < localRangeZ.getBegin() + overlapsZ || gk >= localRangeZ.getEnd() - overlapsZ )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, forLocalBoundary )
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
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
                || gj >= localRangeY.getEnd() || gk < localRangeZ.getBegin() || gk >= localRangeZ.getEnd() )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
         }
      }
   }

   a.setValue( 0 );
   a.getView().forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            if( gi < localRangeX.getBegin() || gi >= localRangeX.getEnd() || gj < localRangeY.getBegin()
                || gj >= localRangeY.getEnd() || gk < localRangeZ.getBegin() || gk >= localRangeZ.getEnd() )
            {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
            }
            else {
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
            }
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, forGhosts )
{
   test_helper_forGhosts( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray, typename BlockType >
void
test_helper_synchronize_D3Q7( DistributedArray& a,
                              int globalSize,
                              int rank,
                              const std::vector< BlockType >& decomposition,
                              const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) = gi + gj + gk;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.setSynchronizationPattern( NDArraySyncPatterns::D3Q7 );
   setNeighbors( s1, NDArraySyncPatterns::D3Q7, rank, decomposition, globalBlock );
   s1.synchronize( a );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            const auto value = a.getElement( gi, gj, gk );
            // handle periodic boundaries
            const IndexType new_gi = ( gi + globalSize ) % globalSize;
            const IndexType new_gj = ( gj + globalSize ) % globalSize;
            const IndexType new_gk = ( gk + globalSize ) % globalSize;
            // calculate the expected value
            IndexType expected_value = new_gi + new_gj + new_gk;
            // corners are skipped in the D3Q7 synchronization pattern
            if( ! ( gi >= localRangeX.getBegin() && gi < localRangeX.getEnd() && gj >= localRangeY.getBegin()
                    && gj < localRangeY.getEnd() )
                && ! ( gi >= localRangeX.getBegin() && gi < localRangeX.getEnd() && gk >= localRangeZ.getBegin()
                       && gk < localRangeZ.getEnd() )
                && ! ( gj >= localRangeY.getBegin() && gj < localRangeY.getEnd() && gk >= localRangeZ.getBegin()
                       && gk < localRangeZ.getEnd() ) )
               expected_value = -1;
            // check the result
            EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }

   a.setValue( -1 );
   a.getView().forAll( setter );
   DistributedNDArraySynchronizer< typename DistributedArray::ViewType > s2;
   s2.setSynchronizationPattern( NDArraySyncPatterns::D3Q7 );
   setNeighbors( s2, NDArraySyncPatterns::D3Q7, rank, decomposition, globalBlock );
   auto view = a.getView();
   s2.synchronize( view );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            const auto value = a.getElement( gi, gj, gk );
            // handle periodic boundaries
            const IndexType new_gi = ( gi + globalSize ) % globalSize;
            const IndexType new_gj = ( gj + globalSize ) % globalSize;
            const IndexType new_gk = ( gk + globalSize ) % globalSize;
            // calculate the expected value
            IndexType expected_value = new_gi + new_gj + new_gk;
            // corners are skipped in the D3Q7 synchronization pattern
            if( ! ( gi >= localRangeX.getBegin() && gi < localRangeX.getEnd() && gj >= localRangeY.getBegin()
                    && gj < localRangeY.getEnd() )
                && ! ( gi >= localRangeX.getBegin() && gi < localRangeX.getEnd() && gk >= localRangeZ.getBegin()
                       && gk < localRangeZ.getEnd() )
                && ! ( gj >= localRangeY.getBegin() && gj < localRangeY.getEnd() && gk >= localRangeZ.getBegin()
                       && gk < localRangeZ.getEnd() ) )
               expected_value = -1;
            // check the result
            EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, synchronize_D3Q7 )
{
   test_helper_synchronize_D3Q7(
      this->distributedNDArray, this->globalSize, this->rank, this->decomposition, this->globalBlock );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray, typename BlockType >
void
test_helper_synchronize_D3Q27( DistributedArray& a,
                               int globalSize,
                               int rank,
                               const std::vector< BlockType >& decomposition,
                               const BlockType& globalBlock )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlapsX = get< 0 >( typename DistributedArray::OverlapsType{} );
   const int overlapsY = get< 1 >( typename DistributedArray::OverlapsType{} );
   const int overlapsZ = get< 2 >( typename DistributedArray::OverlapsType{} );

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();
   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) = gi + gj + gk;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.setSynchronizationPattern( NDArraySyncPatterns::D3Q27 );
   setNeighbors( s1, NDArraySyncPatterns::D3Q27, rank, decomposition, globalBlock );
   s1.synchronize( a );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            const auto value = a.getElement( gi, gj, gk );
            // handle periodic boundaries
            const IndexType new_gi = ( gi + globalSize ) % globalSize;
            const IndexType new_gj = ( gj + globalSize ) % globalSize;
            const IndexType new_gk = ( gk + globalSize ) % globalSize;
            // calculate the expected value
            IndexType expected_value = new_gi + new_gj + new_gk;
            // check the result
            EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }

   a.setValue( -1 );
   a.getView().forAll( setter );
   DistributedNDArraySynchronizer< typename DistributedArray::ViewType > s2;
   s2.setSynchronizationPattern( NDArraySyncPatterns::D3Q27 );
   setNeighbors( s2, NDArraySyncPatterns::D3Q27, rank, decomposition, globalBlock );
   auto view = a.getView();
   s2.synchronize( view );

   for( IndexType gi = localRangeX.getBegin() - overlapsX; gi < localRangeX.getEnd() + overlapsX; gi++ ) {
      for( IndexType gj = localRangeY.getBegin() - overlapsY; gj < localRangeY.getEnd() + overlapsY; gj++ ) {
         for( IndexType gk = localRangeZ.getBegin() - overlapsZ; gk < localRangeZ.getEnd() + overlapsZ; gk++ ) {
            const auto value = a.getElement( gi, gj, gk );
            // handle periodic boundaries
            const IndexType new_gi = ( gi + globalSize ) % globalSize;
            const IndexType new_gj = ( gj + globalSize ) % globalSize;
            const IndexType new_gk = ( gk + globalSize ) % globalSize;
            // calculate the expected value
            IndexType expected_value = new_gi + new_gj + new_gk;
            // check the result
            EXPECT_EQ( value, expected_value ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }
}

TYPED_TEST( DistributedNDArrayOverlaps_3D_test, synchronize_D3Q27 )
{
   test_helper_synchronize_D3Q27(
      this->distributedNDArray, this->globalSize, this->rank, this->decomposition, this->globalBlock );
}

#endif  // HAVE_GTEST

#include "../../main_mpi.h"
