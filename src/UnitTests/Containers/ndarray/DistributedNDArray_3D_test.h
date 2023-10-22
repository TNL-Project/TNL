#ifdef HAVE_GTEST
   #include <gtest/gtest.h>

   #include <TNL/Containers/BlockPartitioning.h>
   #include <TNL/Containers/DistributedNDArray.h>
   #include <TNL/Containers/DistributedNDArrayView.h>
   #include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

/*
 * Light check of DistributedNDArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 31 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArray_3D_test : public ::testing::Test
{
protected:
   using ValueType = typename DistributedNDArray::ValueType;
   using DeviceType = typename DistributedNDArray::DeviceType;
   using IndexType = typename DistributedNDArray::IndexType;
   using DistributedNDArrayType = DistributedNDArray;

   const int globalSize = 31;  // prime number to force non-uniform distribution

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedNDArrayType distributedNDArray;

   const int rank = TNL::MPI::GetRank( communicator );
   const int nproc = TNL::MPI::GetSize( communicator );

   DistributedNDArray_3D_test()
   {
      using BlockType = Block< 3, IndexType >;
      using MultiIndex = typename BlockType::CoordinatesType;

      const BlockType globalBlock = { MultiIndex{ 0, 0, 0 }, MultiIndex{ globalSize, globalSize, globalSize } };
      const std::vector< BlockType > decomposition = decomposeBlockOptimal( globalBlock, nproc );
      const BlockType& localBlock = decomposition.at( rank );

      distributedNDArray.setSizes( globalSize, globalSize, globalSize );
      distributedNDArray.template setDistribution< 0 >( localBlock.begin.x(), localBlock.end.x(), communicator );
      distributedNDArray.template setDistribution< 1 >( localBlock.begin.y(), localBlock.end.y(), communicator );
      distributedNDArray.template setDistribution< 2 >( localBlock.begin.z(), localBlock.end.z(), communicator );
      distributedNDArray.allocate();
   }
};

// types for which DistributedNDArray_3D_test is instantiated
using DistributedNDArrayTypes =
   ::testing::Types< DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0, 0 >,     // X, Y, Z
                                                  std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                                  Devices::Host > >
   #ifdef __CUDACC__
                     ,
                     DistributedNDArray< NDArray< double,
                                                  SizesHolder< int, 0, 0, 0 >,     // X, Y, Z
                                                  std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                                  Devices::Cuda > >
   #endif
                     >;

TYPED_TEST_SUITE( DistributedNDArray_3D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArray_3D_test, checkSumOfLocalSizes )
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
}

TYPED_TEST( DistributedNDArray_3D_test, setLike )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   const auto localRangeX = this->distributedNDArray.template getLocalRange< 0 >();
   const auto localRangeY = this->distributedNDArray.template getLocalRange< 1 >();
   const auto localRangeZ = this->distributedNDArray.template getLocalRange< 2 >();
   const auto localStorageSize = ( localRangeX.getEnd() - localRangeX.getBegin() )
                               * ( localRangeY.getEnd() - localRangeY.getBegin() )
                               * ( localRangeZ.getEnd() - localRangeZ.getBegin() );
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localStorageSize );

   DistributedNDArrayType copy;
   EXPECT_EQ( copy.getLocalStorageSize(), 0 );
   copy.setLike( this->distributedNDArray );
   EXPECT_EQ( copy.template getLocalRange< 0 >(), this->distributedNDArray.template getLocalRange< 0 >() );
   EXPECT_EQ( copy.template getLocalRange< 1 >(), this->distributedNDArray.template getLocalRange< 1 >() );
   EXPECT_EQ( copy.template getLocalRange< 2 >(), this->distributedNDArray.template getLocalRange< 2 >() );
   EXPECT_EQ( copy.getLocalStorageSize(), localStorageSize );
}

TYPED_TEST( DistributedNDArray_3D_test, reset )
{
   const auto localRangeX = this->distributedNDArray.template getLocalRange< 0 >();
   const auto localRangeY = this->distributedNDArray.template getLocalRange< 1 >();
   const auto localRangeZ = this->distributedNDArray.template getLocalRange< 2 >();
   const auto localStorageSize = ( localRangeX.getEnd() - localRangeX.getBegin() )
                               * ( localRangeY.getEnd() - localRangeY.getBegin() )
                               * ( localRangeZ.getEnd() - localRangeZ.getBegin() );
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localStorageSize );

   this->distributedNDArray.reset();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), 0 );
}

TYPED_TEST( DistributedNDArray_3D_test, elementwiseAccess )
{
   using IndexType = typename TestFixture::IndexType;

   this->distributedNDArray.setValue( 0 );
   const auto localRangeX = this->distributedNDArray.template getLocalRange< 0 >();
   const auto localRangeY = this->distributedNDArray.template getLocalRange< 1 >();
   const auto localRangeZ = this->distributedNDArray.template getLocalRange< 2 >();

   // check initial value
   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( this->distributedNDArray.getElement( gi, gj, gk ), 0 );
         }
      }
   }

   // use operator()
   if( std::is_same_v< typename TestFixture::DeviceType, Devices::Host > ) {
      for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
         for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
            for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
               this->distributedNDArray( gi, gj, gk ) = gi + 1;
            }
         }
      }

      // check set value
      for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
         for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
            for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
               EXPECT_EQ( this->distributedNDArray.getElement( gi, gj, gk ), gi + 1 );
               EXPECT_EQ( this->distributedNDArray( gi, gj, gk ), gi + 1 );
            }
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, copyAssignment )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   this->distributedNDArray.setValue( 1 );
   DistributedNDArrayType copy;
   copy = this->distributedNDArray;
   // no binding, but deep copy
   EXPECT_NE( copy.getLocalView().getData(), this->distributedNDArray.getLocalView().getData() );
   EXPECT_EQ( copy.getLocalView(), this->distributedNDArray.getLocalView() );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_comparisonOperators( DistributedArray& u, DistributedArray& v, DistributedArray& w )
{
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;
   using MultiIndex = Containers::StaticArray< 3, IndexType >;

   const auto localRangeX = u.template getLocalRange< 0 >();
   const auto localRangeY = u.template getLocalRange< 1 >();
   const auto localRangeZ = u.template getLocalRange< 2 >();

   auto u_view = u.getLocalView();
   auto v_view = v.getLocalView();
   auto w_view = w.getLocalView();

   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      u_view( i[ 0 ] - localRangeX.getBegin(), i[ 1 ] - localRangeY.getBegin(), i[ 2 ] - localRangeZ.getBegin() ) = i[ 0 ];
      v_view( i[ 0 ] - localRangeX.getBegin(), i[ 1 ] - localRangeY.getBegin(), i[ 2 ] - localRangeZ.getBegin() ) = i[ 0 ];
      w_view( i[ 0 ] - localRangeX.getBegin(), i[ 1 ] - localRangeY.getBegin(), i[ 2 ] - localRangeZ.getBegin() ) = 2 * i[ 0 ];
   };
   Algorithms::parallelFor< DeviceType >( MultiIndex{ localRangeX.getBegin(), localRangeY.getBegin(), localRangeZ.getBegin() },
                                          MultiIndex{ localRangeX.getEnd(), localRangeY.getEnd(), localRangeZ.getEnd() },
                                          kernel );
}

TYPED_TEST( DistributedNDArray_3D_test, comparisonOperators )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   DistributedNDArrayType& u = this->distributedNDArray;
   DistributedNDArrayType v, w;
   v.setLike( u );
   w.setLike( u );

   test_helper_comparisonOperators( u, v, w );

   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forAll( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

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

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
         }
      }
   }

   a.setValue( 0 );
   a.getView().forAll( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forAll )
{
   test_helper_forAll( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forInterior( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();

   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi == 0 || gi == a.template getSize< 0 >() - 1 || gj == 0 || gj == a.template getSize< 1 >() - 1 || gk == 0
                || gk == a.template getSize< 2 >() - 1 )
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
            else
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }

   a.setValue( 0 );
   a.getView().forInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi == 0 || gi == a.template getSize< 0 >() - 1 || gj == 0 || gj == a.template getSize< 1 >() - 1 || gk == 0
                || gk == a.template getSize< 2 >() - 1 )
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
            else
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forInterior )
{
   test_helper_forInterior( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forLocalInterior( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();

   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a.forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
         }
      }
   }

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a.getView().forLocalInterior( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 1 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forLocalInterior )
{
   test_helper_forLocalInterior( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();

   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   a.forBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi == 0 || gi == a.template getSize< 0 >() - 1 || gj == 0 || gj == a.template getSize< 1 >() - 1 || gk == 0
                || gk == a.template getSize< 2 >() - 1 )
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
            else
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }

   a.setValue( 0 );
   a.getView().forBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            if( gi == 0 || gi == a.template getSize< 0 >() - 1 || gj == 0 || gj == a.template getSize< 1 >() - 1 || gk == 0
                || gk == a.template getSize< 2 >() - 1 )
               EXPECT_EQ( a.getElement( gi, gj, gk ), 1 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
            else
               EXPECT_EQ( a.getElement( gi, gj, gk ), 0 ) << "gi = " << gi << ", gj = " << gj << ", gk = " << gk;
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forBoundary )
{
   test_helper_forBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void
test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();

   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
         }
      }
   }

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.getView().forLocalBoundary( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forLocalBoundary )
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

   const auto localRangeX = a.template getLocalRange< 0 >();
   const auto localRangeY = a.template getLocalRange< 1 >();
   const auto localRangeZ = a.template getLocalRange< 2 >();

   auto a_view = a.getLocalView();

   auto setter = [ = ] __cuda_callable__( IndexType gi, IndexType gj, IndexType gk ) mutable
   {
      a_view( gi - localRangeX.getBegin(), gj - localRangeY.getBegin(), gk - localRangeZ.getBegin() ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
         }
      }
   }

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.getView().forGhosts( setter );

   for( IndexType gi = localRangeX.getBegin(); gi < localRangeX.getEnd(); gi++ ) {
      for( IndexType gj = localRangeY.getBegin(); gj < localRangeY.getEnd(); gj++ ) {
         for( IndexType gk = localRangeZ.getBegin(); gk < localRangeZ.getEnd(); gk++ ) {
            EXPECT_EQ( a.getElement( gi, gj, gk ), 0 );
         }
      }
   }
}

TYPED_TEST( DistributedNDArray_3D_test, forGhosts )
{
   test_helper_forGhosts( this->distributedNDArray );
}

#endif  // HAVE_GTEST

#include "../../main_mpi.h"
