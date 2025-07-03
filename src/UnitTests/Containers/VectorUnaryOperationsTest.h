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

namespace unary_tests {

// prime number to force non-uniform distribution in block-wise algorithms
constexpr int VECTOR_TEST_SIZE = 97;

// test fixture for typed tests
template< typename T >
class VectorUnaryOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
#ifdef STATIC_VECTOR
   template< typename Real >
   using Vector = StaticVector< VectorOrView::getSize(), Real >;
#else
   using NonConstReal = std::remove_const_t< typename VectorOrView::ValueType >;
   #ifdef DISTRIBUTED_VECTOR
   using VectorType = DistributedVector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   template< typename Real >
   using Vector = DistributedVector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;

   const MPI_Comm communicator = MPI_COMM_WORLD;

   const int rank = GetRank( communicator );
   const int nproc = GetSize( communicator );

   // some arbitrary even value (but must be 0 if not distributed)
   const int ghosts = ( nproc > 1 ) ? 4 : 0;
   #else
   using VectorType = Containers::Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   template< typename Real >
   using Vector = Containers::Vector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   #endif
#endif
};

#if defined( __CUDACC__ ) || defined( __HIP__ )
using TestDevice = Devices::GPU;
#else
using TestDevice = Devices::Host;
#endif

#if defined( COMPLEX_VALUE_TYPE )
   #if defined( __CUDACC__ ) || defined( __HIP__ )
using TestValueType = TNL::Arithmetics::Complex< float >;
      // some functions are not defined for TNL::Arithmetics::Complex
      #define TNL_COMPLEX
   #else
using TestValueType = std::complex< double >;
   #endif
#else
using TestValueType = double;
#endif

// types for which VectorUnaryOperationsTest is instantiated
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
   VectorView< StaticVector< 3, TestValueType >, TestDevice >,
   VectorView< StaticVector< 3, CustomScalar< double > >, TestDevice > >;
   #else
using VectorTypes = ::testing::Types<  //
   Vector< TestValueType, TestDevice >,
   VectorView< TestValueType, TestDevice >,
   VectorView< const TestValueType, TestDevice >,
   Vector< CustomScalar< double >, TestDevice >,
   VectorView< CustomScalar< double >, TestDevice > >;
   #endif
#endif

TYPED_TEST_SUITE( VectorUnaryOperationsTest, VectorTypes );

#define EXPECTED_VECTOR( TestFixture, function )                 \
   using ExpectedVector = typename TestFixture::template Vector< \
      Expressions::RemoveET< decltype( function( typename VectorOrView::ValueType{} ) ) > >;

#ifdef STATIC_VECTOR
   #define SETUP_UNARY_VECTOR_TEST( _ )                        \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using ValueType = typename VectorOrView::ValueType;      \
                                                               \
      VectorOrView V1;                                         \
      VectorOrView V2;                                         \
                                                               \
      V1 = ValueType( 1 );                                     \
      V2 = ValueType( 2 );                                     \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _, begin, end, function ) \
      using VectorOrView = typename TestFixture::VectorOrView;         \
      using ValueType = typename VectorOrView::ValueType;              \
      EXPECTED_VECTOR( TestFixture, function );                        \
      constexpr int _size = VectorOrView::getSize();                   \
                                                                       \
      VectorOrView V1;                                                 \
      ExpectedVector expected;                                         \
                                                                       \
      const double h = (double) ( end - begin ) / _size;               \
      for( int i = 0; i < _size; i++ ) {                               \
         const ValueType x = begin + ValueType( i * h );               \
         V1[ i ] = x;                                                  \
         expected[ i ] = function( x );                                \
      }                                                                \
      (void) 0  // dummy statement here enforces ';' after the macro use

#elif defined( DISTRIBUTED_VECTOR )
   #define SETUP_UNARY_VECTOR_TEST( size )                                                                          \
      using VectorType = typename TestFixture::VectorType;                                                          \
      using VectorOrView = typename TestFixture::VectorOrView;                                                      \
      using ValueType = typename VectorType::ValueType;                                                             \
      using LocalRangeType = typename VectorOrView::LocalRangeType;                                                 \
      const LocalRangeType localRange = splitRange< typename VectorOrView::IndexType >( size, this->communicator ); \
      using Synchronizer = DistributedArraySynchronizer< VectorOrView >;                                            \
                                                                                                                    \
      VectorType _V1;                                                                                               \
      VectorType _V2;                                                                                               \
      _V1.setDistribution( localRange, this->ghosts, size, this->communicator );                                    \
      _V2.setDistribution( localRange, this->ghosts, size, this->communicator );                                    \
                                                                                                                    \
      auto _synchronizer = std::make_shared< Synchronizer >( localRange, this->ghosts / 2, this->communicator );    \
      _V1.setSynchronizer( _synchronizer );                                                                         \
      _V2.setSynchronizer( _synchronizer );                                                                         \
                                                                                                                    \
      _V1 = ValueType( 1 );                                                                                         \
      _V2 = ValueType( 2 );                                                                                         \
                                                                                                                    \
      VectorOrView V1( _V1 );                                                                                       \
      VectorOrView V2( _V2 );                                                                                       \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function )                                                     \
      using VectorType = typename TestFixture::VectorType;                                                                    \
      using VectorOrView = typename TestFixture::VectorOrView;                                                                \
      using ValueType = typename VectorType::ValueType;                                                                       \
      EXPECTED_VECTOR( TestFixture, function );                                                                               \
      using HostVector = typename VectorType::template Self< ValueType, Devices::Host >;                                      \
      using HostExpectedVector = typename ExpectedVector::template Self< typename ExpectedVector::ValueType, Devices::Host >; \
      using LocalRangeType = typename VectorOrView::LocalRangeType;                                                           \
      const LocalRangeType localRange = splitRange< typename VectorOrView::IndexType >( size, this->communicator );           \
      using Synchronizer = DistributedArraySynchronizer< VectorOrView >;                                                      \
                                                                                                                              \
      HostVector _V1h;                                                                                                        \
      HostExpectedVector expected_h;                                                                                          \
      _V1h.setDistribution( localRange, this->ghosts, size, this->communicator );                                             \
      expected_h.setDistribution( localRange, this->ghosts, size, this->communicator );                                       \
                                                                                                                              \
      const double h = (double) ( end - begin ) / size;                                                                       \
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ ) {                                                    \
         const ValueType x = begin + ValueType( i * h );                                                                      \
         _V1h[ i ] = x;                                                                                                       \
         expected_h[ i ] = function( x );                                                                                     \
      }                                                                                                                       \
      for( int i = localRange.getSize(); i < _V1h.getLocalView().getSize(); i++ )                                             \
         _V1h.getLocalView()[ i ] = expected_h.getLocalView()[ i ] = 0;                                                       \
                                                                                                                              \
      VectorType _V1;                                                                                                         \
      _V1 = _V1h;                                                                                                             \
      VectorOrView V1( _V1 );                                                                                                 \
      ExpectedVector expected;                                                                                                \
      expected = expected_h;                                                                                                  \
                                                                                                                              \
      auto _synchronizer = std::make_shared< Synchronizer >( localRange, this->ghosts / 2, this->communicator );              \
      _V1.setSynchronizer( _synchronizer );                                                                                   \
      expected.setSynchronizer( _synchronizer );                                                                              \
      expected.startSynchronization();                                                                                        \
      (void) 0  // dummy statement here enforces ';' after the macro use

#else
   #define SETUP_UNARY_VECTOR_TEST( size )                     \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using ValueType = typename VectorType::ValueType;        \
                                                               \
      VectorType _V1( size );                                  \
      VectorType _V2( size );                                  \
                                                               \
      _V1 = ValueType( 1 );                                    \
      _V2 = ValueType( 2 );                                    \
                                                               \
      VectorOrView V1( _V1 );                                  \
      VectorOrView V2( _V2 );                                  \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function )                                                     \
      using VectorType = typename TestFixture::VectorType;                                                                    \
      using VectorOrView = typename TestFixture::VectorOrView;                                                                \
      using ValueType = typename VectorType::ValueType;                                                                       \
      EXPECTED_VECTOR( TestFixture, function );                                                                               \
      using HostVector = typename VectorType::template Self< ValueType, Devices::Host >;                                      \
      using HostExpectedVector = typename ExpectedVector::template Self< typename ExpectedVector::ValueType, Devices::Host >; \
                                                                                                                              \
      HostVector _V1h( size );                                                                                                \
      HostExpectedVector expected_h( size );                                                                                  \
                                                                                                                              \
      const double h = (double) ( end - begin ) / size;                                                                       \
      for( int i = 0; i < size; i++ ) {                                                                                       \
         const ValueType x = ValueType( begin ) + ValueType( i * h );                                                         \
         _V1h[ i ] = x;                                                                                                       \
         expected_h[ i ] = function( x );                                                                                     \
      }                                                                                                                       \
                                                                                                                              \
      VectorType _V1;                                                                                                         \
      _V1 = _V1h;                                                                                                             \
      VectorOrView V1( _V1 );                                                                                                 \
      ExpectedVector expected;                                                                                                \
      expected = expected_h;                                                                                                  \
      (void) 0  // dummy statement here enforces ';' after the macro use

#endif

// This is because exact comparison does not work due to rounding errors:
// - the "expected" vector is computed sequentially on CPU
// - the host compiler might decide to use a vectorized version of the
//   math function, which may have slightly different precision
// - GPU may have different precision than CPU, so exact comparison with
//   the result from host is not possible
template< typename Left, typename Right >
void
expect_vectors_near( const Left& _v1, const Right& _v2 )
{
   ASSERT_EQ( _v1.getSize(), _v2.getSize() );
#ifdef STATIC_VECTOR
   for( int i = 0; i < _v1.getSize(); i++ )
      expect_near( _v1[ i ], _v2[ i ], 1e-6 );
#else
   using LeftNonConstReal = Expressions::RemoveET< std::remove_const_t< typename Left::ValueType > >;
   using RightNonConstReal = Expressions::RemoveET< std::remove_const_t< typename Right::ValueType > >;
   #ifdef DISTRIBUTED_VECTOR
   using LeftVector = DistributedVector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = DistributedVector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
   #else
   using LeftVector = Vector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = Vector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
   #endif
   using LeftHostVector = typename LeftVector::template Self< LeftNonConstReal, Devices::Sequential >;
   using RightHostVector = typename RightVector::template Self< RightNonConstReal, Devices::Sequential >;

   // first evaluate expressions
   LeftVector v1;
   v1 = _v1;
   RightVector v2;
   v2 = _v2;
   // then copy to host
   LeftHostVector v1_h;
   v1_h = v1;
   RightHostVector v2_h;
   v2_h = v2;
   #ifdef DISTRIBUTED_VECTOR
   const auto localRange = v1.getLocalRange();
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
   #else
   for( int i = 0; i < v1.getSize(); i++ )
   #endif
      expect_near( v1_h[ i ], v2_h[ i ], 1e-6 );
#endif
}

TYPED_TEST( VectorUnaryOperationsTest, plus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( +V1, ValueType( 1 ) );
   // unary expression
   EXPECT_EQ( +( +V2 ), ValueType( 2 ) );
   // binary expression
   EXPECT_EQ( +( V1 + V1 ), ValueType( 2 ) );
}

TYPED_TEST( VectorUnaryOperationsTest, minus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( -V1, ValueType( -1 ) );
   // unary expression
   EXPECT_EQ( -( -V2 ), ValueType( 2 ) );
   // binary expression
   EXPECT_EQ( -( V1 + V1 ), ValueType( -2 ) );
}

// operation does not make sense for complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, logicalNot )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( ! V1, ValueType( 0 ) );
   // unary expression
   EXPECT_EQ( ! ( ! V2 ), ValueType( 1 ) );
   // binary expression
   EXPECT_EQ( ! ( V1 + V1 ), ValueType( 0 ) );
}
#endif

// operation does not make sense for complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, bitNot )
{
   // binary negation is defined only for integral types
   using ValueType = typename TestFixture::VectorOrView::ValueType;
   if constexpr( std::is_integral_v< ValueType > ) {
      SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

      // vector or view
      EXPECT_EQ( ~V1, ~static_cast< ValueType >( 1 ) );
      // unary expression
      EXPECT_EQ( ~( ~V2 ), ValueType( 2 ) );
      // binary expression
      EXPECT_EQ( ~( V1 + V1 ), ~static_cast< ValueType >( 2 ) );
   }
}
#endif

TYPED_TEST( VectorUnaryOperationsTest, abs )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( abs( V1 ), V1 );
   // unary expression
   EXPECT_EQ( abs( -V1 ), V1 );
   // binary expression
   EXPECT_EQ( abs( -V1 - V1 ), V2 );
}

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, sin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sin );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( sin( V1 ), expected );
   // binary expression
   expect_vectors_near( sin( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( sin( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, asin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::asin );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( asin( V1 ), expected );
   // binary expression
   expect_vectors_near( asin( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( asin( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, cos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cos );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( cos( V1 ), expected );
   // binary expression
   expect_vectors_near( cos( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( cos( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, acos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::acos );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( acos( V1 ), expected );
   // binary expression
   expect_vectors_near( acos( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( acos( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, tan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.5, 1.5, TNL::tan );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( tan( V1 ), expected );
   // binary expression
   expect_vectors_near( tan( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( tan( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, atan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::atan );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( atan( V1 ), expected );
   // binary expression
   expect_vectors_near( atan( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( atan( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, sinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::sinh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( sinh( V1 ), expected );
   // binary expression
   expect_vectors_near( sinh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( sinh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, asinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::asinh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( asinh( V1 ), expected );
   // binary expression
   expect_vectors_near( asinh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( asinh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, cosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::cosh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( cosh( V1 ), expected );
   // binary expression
   expect_vectors_near( cosh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( cosh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, acosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::acosh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( acosh( V1 ), expected );
   // binary expression
   expect_vectors_near( acosh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( acosh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, tanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::tanh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( tanh( V1 ), expected );
   // binary expression
   expect_vectors_near( tanh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( tanh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, atanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -0.99, 0.99, TNL::atanh );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( atanh( V1 ), expected );
   // binary expression
   expect_vectors_near( atanh( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( atanh( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, pow )
{
   // FIXME: for integer exponent, the test fails with CUDA
   auto pow3 = []( const auto& i )
   {
      using TNL::pow;
      return pow( i, 3.0 );
   };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, pow3 );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( pow( V1, 3.0 ), expected );
   // binary expression
   expect_vectors_near( pow( two * V1 - V1, 3.0 ), expected );
   // unary expression
   expect_vectors_near( pow( -( -V1 ), 3.0 ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, exp )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::exp );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( exp( V1 ), expected );
   // binary expression
   expect_vectors_near( exp( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( exp( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, sqrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 0, VECTOR_TEST_SIZE, TNL::sqrt );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( sqrt( V1 ), expected );
   // binary expression
   expect_vectors_near( sqrt( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( sqrt( -( -V1 ) ), expected );
}
#endif

// std::cbrt is not defined for std::complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, cbrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cbrt );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( cbrt( V1 ), expected );
   // binary expression
   expect_vectors_near( cbrt( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( cbrt( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, log )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( log( V1 ), expected );
   // binary expression
   expect_vectors_near( log( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( log( -( -V1 ) ), expected );
}
#endif

// function is not defined for TNL::Arithmetics::Complex
#if ! defined( TNL_COMPLEX )
TYPED_TEST( VectorUnaryOperationsTest, log10 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log10 );

   #ifdef COMPLEX_VALUE_TYPE
   const ValueType two = 2;
   #else
   const int two = 2;
   #endif

   // vector or view
   expect_vectors_near( log10( V1 ), expected );
   // binary expression
   expect_vectors_near( log10( two * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( log10( -( -V1 ) ), expected );
}
#endif

// std::log2 is not defined for std::complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, log2 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log2 );

   // vector or view
   expect_vectors_near( log2( V1 ), expected );
   // binary expression
   expect_vectors_near( log2( 2 * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( log2( -( -V1 ) ), expected );
}
#endif

// std::floor is not defined for std::complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, floor )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::floor );

   // vector or view
   expect_vectors_near( floor( V1 ), expected );
   // binary expression
   expect_vectors_near( floor( 2 * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( floor( -( -V1 ) ), expected );
}
#endif

// std::ceil is not defined for std::complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, ceil )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::ceil );

   // vector or view
   expect_vectors_near( ceil( V1 ), expected );
   // binary expression
   expect_vectors_near( ceil( 2 * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( ceil( -( -V1 ) ), expected );
}
#endif

// TNL::sign does not make sense for complex
#if ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, sign )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sign );

   // vector or view
   expect_vectors_near( sign( V1 ), expected );
   // binary expression
   expect_vectors_near( sign( 2 * V1 - V1 ), expected );
   // unary expression
   expect_vectors_near( sign( -( -V1 ) ), expected );
}
#endif

// This test is not suitable for vector-of-static-vectors where the ValueType cannot be cast to bool.
#if ! defined( VECTOR_OF_STATIC_VECTORS ) && ! defined( COMPLEX_VALUE_TYPE )
TYPED_TEST( VectorUnaryOperationsTest, cast )
{
   auto identity = []( auto i )
   {
      return i;
   };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, identity );

   // vector or vector view
   auto expression1 = cast< bool >( V1 );
   static_assert( std::is_same_v< typename decltype( expression1 )::ValueType, bool >,
                  "BUG: the cast function does not work for vector or vector view." );
   EXPECT_EQ( expression1, true );

   // binary expression
   auto expression2 = cast< bool >( V1 + V1 );
   static_assert( std::is_same_v< typename decltype( expression2 )::ValueType, bool >,
                  "BUG: the cast function does not work for binary expression." );
   // FIXME: expression2 cannot be reused, because expression templates for StaticVector and DistributedVector contain
   // references and the test would crash in Release
   //   EXPECT_EQ( expression2, true );
   EXPECT_EQ( cast< bool >( V1 + V1 ), true );

   // unary expression
   auto expression3 = cast< bool >( -V1 );
   static_assert( std::is_same_v< typename decltype( expression3 )::ValueType, bool >,
                  "BUG: the cast function does not work for unary expression." );
   // FIXME: expression3 cannot be reused, because expression templates for StaticVector and DistributedVector contain
   // references and the test would crash in Release
   //   EXPECT_EQ( expression3, true );
   EXPECT_EQ( cast< bool >( -V1 ), true );
}
#endif

}  // namespace unary_tests
