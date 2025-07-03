#pragma once

#if defined( DISTRIBUTED_VECTOR )
   #include <TNL/Containers/DistributedVector.h>
   #include <TNL/Containers/DistributedVectorView.h>
   #include <TNL/Containers/DistributedArraySynchronizer.h>
   #include <TNL/Containers/BlockPartitioning.h>
using namespace TNL::MPI;
#elif defined( STATIC_VECTOR )
   #include <TNL/Containers/StaticVector.h>
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
      #include <TNL/Containers/StaticVector.h>
   #endif
   #include <TNL/Containers/Vector.h>
   #include <TNL/Containers/VectorView.h>
#endif

#include "VectorHelperFunctions.h"
#include "../CustomScalar.h"
#include <TNL/Arithmetics/Complex.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

namespace vertical_tests {

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_REDUCTION_SIZE = 4999;

// test fixture for typed tests
template< typename T >
class VectorVerticalOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
#ifdef STATIC_VECTOR
   template< typename Real >
   using Vector = StaticVector< VectorOrView::getSize(), Real >;
#else
   using NonConstReal = std::remove_const_t< typename VectorOrView::RealType >;
   #ifdef DISTRIBUTED_VECTOR
   using VectorType = DistributedVector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   template< typename Real >
   using Vector = DistributedVector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;

   const MPI_Comm communicator = MPI_COMM_WORLD;

   const int rank = GetRank( communicator );
   const int nproc = GetSize( communicator );

   // some arbitrary value (but must be 0 if not distributed)
   const int ghosts = ( nproc > 1 ) ? 4 : 0;
   #else
   using VectorType = Containers::Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   template< typename Real >
   using Vector = Containers::Vector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   #endif
#endif

   VectorOrView V1;

#ifndef STATIC_VECTOR
   VectorType _V1;
#endif

   void
   reset( int size )
   {
#ifdef STATIC_VECTOR
      setLinearSequence( V1 );
#else
   #ifdef DISTRIBUTED_VECTOR
      using LocalRangeType = typename VectorOrView::LocalRangeType;
      using Synchronizer = DistributedArraySynchronizer< VectorOrView >;
      const LocalRangeType localRange = splitRange< typename VectorOrView::IndexType >( size, communicator );
      _V1.setDistribution( localRange, ghosts, size, communicator );
      _V1.setSynchronizer( std::make_shared< Synchronizer >( localRange, ghosts / 2, communicator ) );
   #else
      _V1.setSize( size );
   #endif
      setLinearSequence( _V1 );
      bindOrAssign( V1, _V1 );
#endif
   }

   VectorVerticalOperationsTest()
   {
      reset( VECTOR_TEST_REDUCTION_SIZE );
   }
};

#define SETUP_VERTICAL_TEST_ALIASES                         \
   using VectorOrView = typename TestFixture::VectorOrView; \
   VectorOrView& V1 = this->V1;                             \
   const int size = V1.getSize();                           \
   (void) 0  // dummy statement here enforces ';' after the macro use

#if defined( __CUDACC__ ) || defined( __HIP__ )
using TestDevice = Devices::GPU;
#else
using TestDevice = Devices::Host;
#endif

#if defined( COMPLEX_VALUE_TYPE )
   #if defined( __CUDACC__ ) || defined( __HIP__ )
using TestValueType = TNL::Arithmetics::Complex< float >;
   #else
using TestValueType = std::complex< float >;
   #endif
#else
using TestValueType = double;
#endif

// types for which VectorVerticalOperationsTest is instantiated
#if defined( DISTRIBUTED_VECTOR )
using VectorTypes = ::testing::Types<  //
   DistributedVector< TestValueType, TestDevice >,
   DistributedVectorView< TestValueType, TestDevice >,
   DistributedVectorView< const TestValueType, TestDevice >,
   DistributedVector< CustomScalar< double >, TestDevice > >;
#elif defined( STATIC_VECTOR )
   #ifdef VECTOR_OF_STATIC_VECTORS
using VectorTypes = ::testing::Types<  //
   StaticVector< 1, StaticVector< 3, TestValueType > >,
   StaticVector< 2, StaticVector< 3, TestValueType > >,
   StaticVector< 3, StaticVector< 3, TestValueType > >,
   StaticVector< 4, StaticVector< 3, TestValueType > >,
   StaticVector< 5, StaticVector< 3, CustomScalar< double > > > >;
   #else
using VectorTypes = ::testing::Types<  //
   StaticVector< 1, TestValueType >,
   StaticVector< 2, TestValueType >,
   StaticVector< 3, TestValueType >,
   StaticVector< 4, TestValueType >,
   StaticVector< 5, CustomScalar< double > > >;
   #endif
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
using VectorTypes = ::testing::Types<  //
   Vector< StaticVector< 3, TestValueType >, TestDevice >,
   VectorView< StaticVector< 3, TestValueType >, TestDevice > >;
   #else
using VectorTypes = ::testing::Types<  //
   Vector< TestValueType, TestDevice >,
   VectorView< TestValueType, TestDevice >,
   VectorView< const TestValueType, TestDevice >,
   Vector< CustomScalar< double >, TestDevice >,
   VectorView< CustomScalar< double >, TestDevice > >;
   #endif
#endif

TYPED_TEST_SUITE( VectorVerticalOperationsTest, VectorTypes );

// FIXME: function does not work for nested vectors - std::numeric_limits does not make sense for vector types
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, max )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( max( V1 ), size - 1 );
   // unary expression
   EXPECT_EQ( max( -V1 ), 0 );
   // binary expression
   EXPECT_EQ( max( V1 + 2 ), size - 1 + 2 );
}
#endif

// FIXME: function does not work for nested vectors - the reduction operation expects a scalar type
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, argMax )
{
   SETUP_VERTICAL_TEST_ALIASES;
   using RealType = typename TestFixture::VectorOrView::RealType;

   // vector or view
   EXPECT_EQ( argMax( V1 ), std::make_pair( (RealType) size - 1, size - 1 ) );
   // unary expression
   EXPECT_EQ( argMax( -V1 ), std::make_pair( (RealType) 0, 0 ) );
   // expression
   EXPECT_EQ( argMax( V1 + 2 ), std::make_pair( (RealType) size - 1 + 2, size - 1 ) );
}
#endif

// FIXME: function does not work for nested vectors - std::numeric_limits does not make sense for vector types
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, min )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( min( V1 ), 0 );
   // unary expression
   EXPECT_EQ( min( -V1 ), 1 - size );
   // binary expression
   EXPECT_EQ( min( V1 + 2 ), 2 );
}
#endif

// FIXME: function does not work for nested vectors - the reduction operation expects a scalar type
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, argMin )
{
   SETUP_VERTICAL_TEST_ALIASES;
   using RealType = typename TestFixture::VectorOrView::RealType;

   // vector or view
   EXPECT_EQ( argMin( V1 ), std::make_pair( (RealType) 0, 0 ) );
   // unary expression
   EXPECT_EQ( argMin( -V1 ), std::make_pair( (RealType) 1 - size, size - 1 ) );
   // binary expression
   EXPECT_EQ( argMin( V1 + 2 ), std::make_pair( (RealType) 2, 0 ) );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, sum )
{
   SETUP_VERTICAL_TEST_ALIASES;

#ifdef COMPLEX_VALUE_TYPE
   using ValueType = typename TestFixture::VectorOrView::ValueType;
   const ValueType one = 1;
   const ValueType expected = 0.5 * size * ( size - 1 );
   const ValueType expected_2 = 0.5 * size * ( size - 1 ) - size;
#else
   const int one = 1;
   const auto expected = 0.5 * size * ( size - 1 );
   const auto expected_2 = 0.5 * size * ( size - 1 ) - size;
#endif

   // vector or view
   EXPECT_EQ( sum( V1 ), expected );
   // unary expression
   EXPECT_EQ( sum( -V1 ), -expected );
   // binary expression
   EXPECT_EQ( sum( V1 - one ), expected_2 );
}

// FIXME: function does not work for nested vectors - max does not work for nested vectors
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, maxNorm )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( maxNorm( V1 ), size - 1 );
   // unary expression
   EXPECT_EQ( maxNorm( -V1 ), size - 1 );
   // binary expression
   EXPECT_EQ( maxNorm( V1 - size ), size );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, l1Norm )
{
#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
#endif
   const int size = V1.getSize();

#ifdef COMPLEX_VALUE_TYPE
   using ValueType = typename TestFixture::VectorOrView::ValueType;
   const ValueType two = 2;
#else
   const int two = 2;
#endif

   // vector or vector view
   EXPECT_EQ( l1Norm( V1 ), size );
   // unary expression
   EXPECT_EQ( l1Norm( -V1 ), size );
   // binary expression
   EXPECT_EQ( l1Norm( two * V1 - V1 ), size );
}

// FIXME: l2Norm does not work for nested vectors - dangling references due to Static*ExpressionTemplate
//        classes binding to temporary objects which get destroyed before l2Norm returns
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, l2Norm )
{
   #ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif
   const int size = V1.getSize();

   using ValueType = typename TestFixture::VectorOrView::ValueType;
   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   const auto expected = std::sqrt( size );

   auto epsilon = std::numeric_limits< double >::epsilon();
   if constexpr( is_complex_v< ValueType > )
      epsilon = 100 * std::numeric_limits< typename ValueType::value_type >::epsilon();

   // vector or vector view
   expect_near( l2Norm( V1 ), expected, epsilon );
   // unary expression
   expect_near( l2Norm( -V1 ), expected, epsilon );
   // binary expression
   expect_near( l2Norm( two * V1 - V1 ), expected, epsilon );
}
#endif

// FIXME function does not work for nested vectors - compilation error
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, lpNorm )
{
   #ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif
   const int size = V1.getSize();

   using ValueType = typename TestFixture::VectorOrView::ValueType;
   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   const auto expectedL1norm = size;
   const auto expectedL2norm = std::sqrt( size );
   const auto expectedL3norm = std::cbrt( size );

   auto epsilon = 64 * std::numeric_limits< decltype( expectedL3norm ) >::epsilon();
   if constexpr( is_complex_v< ValueType > )
      epsilon = 100 * std::numeric_limits< typename ValueType::value_type >::epsilon();

   // vector or vector view
   EXPECT_EQ( lpNorm( V1, 1.0 ), expectedL1norm );
   expect_near( lpNorm( V1, 2.0 ), expectedL2norm, epsilon );
   expect_near( lpNorm( V1, 3.0 ), expectedL3norm, epsilon );
   // unary expression
   EXPECT_EQ( lpNorm( -V1, 1.0 ), expectedL1norm );
   expect_near( lpNorm( -V1, 2.0 ), expectedL2norm, epsilon );
   expect_near( lpNorm( -V1, 3.0 ), expectedL3norm, epsilon );
   // binary expression
   EXPECT_EQ( lpNorm( two * V1 - V1, 1.0 ), expectedL1norm );
   expect_near( lpNorm( two * V1 - V1, 2.0 ), expectedL2norm, epsilon );
   expect_near( lpNorm( two * V1 - V1, 3.0 ), expectedL3norm, epsilon );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, product )
{
   // VERY small size to avoid overflows
   this->reset( 16 );

#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 2 );
   const typename TestFixture::VectorOrView& V2( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 2 );
   const typename TestFixture::VectorOrView& V2( this->_V1 );
#endif
   const int size = V2.getSize();

#ifdef COMPLEX_VALUE_TYPE
   using ValueType = typename TestFixture::VectorOrView::ValueType;
   const ValueType two = 2;
   const ValueType expected = std::exp2( size );
   const ValueType expected_2 = std::exp2( size ) * ( ( size % 2 ) ? -1 : 1 );
#else
   const int two = 2;
   const auto expected = std::exp2( size );
   const auto expected_2 = std::exp2( size ) * ( ( size % 2 != 0 ) ? -1 : 1 );
#endif

   // vector or vector view
   EXPECT_EQ( product( V2 ), expected );
   // unary expression
   EXPECT_EQ( product( -V2 ), expected_2 );
   // binary expression
   EXPECT_EQ( product( two * V2 - V2 ), expected );
}

// StaticVector and complex are not contextually convertible to bool
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorVerticalOperationsTest, all_const_ones )
{
   #ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif

   // vector or vector view
   EXPECT_TRUE( all( V1 ) );
   // unary expression
   EXPECT_TRUE( all( -V1 ) );
   // binary expression
   EXPECT_TRUE( all( V1 + V1 ) );
}

TYPED_TEST( VectorVerticalOperationsTest, all_linear )
{
   #ifdef STATIC_VECTOR
   setLinearSequence( this->V1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setLinearSequence( this->_V1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif

   // vector or vector view
   EXPECT_FALSE( all( V1 ) );
   // unary expression
   EXPECT_FALSE( all( -V1 ) );
   // binary expression
   EXPECT_FALSE( all( V1 + V1 ) );
}

TYPED_TEST( VectorVerticalOperationsTest, any_const_zeros )
{
   #ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 0 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 0 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif

   // vector or vector view
   EXPECT_FALSE( any( V1 ) );
   // unary expression
   EXPECT_FALSE( any( -V1 ) );
   // binary expression
   EXPECT_FALSE( any( V1 + V1 ) );
}

TYPED_TEST( VectorVerticalOperationsTest, any_linear )
{
   #ifdef STATIC_VECTOR
   setLinearSequence( this->V1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   #else
   // we have to use _V1 because V1 might be a const view
   setLinearSequence( this->_V1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   #endif

   if( V1.getSize() > 1 ) {
      // vector or vector view
      EXPECT_TRUE( any( V1 ) );
      // unary expression
      EXPECT_TRUE( any( -V1 ) );
      // binary expression
      EXPECT_TRUE( any( V1 + V1 ) );
   }
   else {
      // vector or vector view
      EXPECT_FALSE( any( V1 ) );
      // unary expression
      EXPECT_FALSE( any( -V1 ) );
      // binary expression
      EXPECT_FALSE( any( V1 + V1 ) );
   }
}

TYPED_TEST( VectorVerticalOperationsTest, argAny )
{
   using Index = typename TestFixture::VectorOrView::IndexType;
   #ifdef STATIC_VECTOR
   setLinearSequence( this->V1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
   const Index step = 1;
   #else
   // we have to use _V1 because V1 might be a const view
   setLinearSequence( this->_V1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
   const Index step = VECTOR_TEST_REDUCTION_SIZE / 5;
   #endif

   using Index = typename TestFixture::VectorOrView::IndexType;
   for( Index i = 0; i < V1.getSize(); i += step ) {
      auto [ check, idx ] = argAny( greaterEqual( V1, i ) );
      EXPECT_TRUE( check ) << "i = " << i;
      EXPECT_EQ( idx, i ) << "i = " << i;
   }
   auto [ check, idx ] = argAny( greaterEqual( V1, V1.getSize() ) );
   EXPECT_FALSE( check );
}

#endif

}  // namespace vertical_tests
