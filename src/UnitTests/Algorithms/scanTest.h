#pragma once

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Arithmetics/experimental/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/scan.h>
#include "../CustomScalar.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics::experimental;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::detail;

// this is a workaround for an nvcc 11.7 bug: it drops the scope of enum class members in template function calls
#ifdef __CUDACC__
static constexpr auto Inclusive = TNL::Algorithms::detail::ScanType::Inclusive;
static constexpr auto Exclusive = TNL::Algorithms::detail::ScanType::Exclusive;
#endif

// test fixture for typed tests
template< typename Array >
class ScanTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using ArrayView = Containers::ArrayView< ValueType, DeviceType, IndexType >;
   using VectorView = Containers::VectorView< ValueType, DeviceType, IndexType >;
   using HostArrayType = typename ArrayType::template Self< ValueType, Devices::Sequential >;

   ArrayType a, b, c;
   ArrayView a_view, b_view, c_view;
   VectorView av_view, bv_view, cv_view;
   HostArrayType array_host, input_host, expected_host;

   // should be small enough to have fast tests, but larger than minGPUReductionDataSize
   // and large enough to require multiple CUDA blocks for reduction
   // also should be a prime number to cause non-uniform distribution of the work
   const int size = 9377;

   ScanTest()
   {
      resetWorkingArrays();
      input_host = expected_host = a;
   }

   void
   resetWorkingArrays()
   {
      a.setSize( size );
      a.setValue( -1 );
      c = b = a;
      a_view.bind( a );
      b_view.bind( b );
      c_view.bind( c );
      av_view.bind( a );
      bv_view.bind( b );
      cv_view.bind( c );

      // make sure that we perform tests with multiple CUDA grids
      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::maxGridSize() = 3;
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::maxGridSize() = 3;
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::maxGridSize() = 3;
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::maxGridSize() = 3;
      }
   }

   template< Algorithms::detail::ScanType ScanType >
   void
   checkResult( const ArrayType& array )
   {
      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
         // skip the check for too small arrays
         if( array.getSize() > 256 ) {
            // we don't care which kernel launcher was actually used
            const auto gridsCount = TNL::max(
               GPUScanKernelLauncher< ScanType, ScanPhaseType::WriteInFirstPhase, ValueType >::gridsCount().load(),
               GPUScanKernelLauncher< ScanType, ScanPhaseType::WriteInSecondPhase, ValueType >::gridsCount().load() );
            EXPECT_GT( gridsCount, 1 );
         }
      }

      array_host = array;

      for( int i = 0; i < array.getSize(); i++ )
         EXPECT_EQ( array_host[ i ], expected_host[ i ] ) << "arrays differ at index i = " << i;
   }
};

// types for which ScanTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Array< CustomScalar< int >, Devices::Sequential, short >,
   Array< int, Devices::Sequential, short >,
   Array< long, Devices::Sequential, short >,
   Array< double, Devices::Sequential, short >,
   //Array< Quad< float >,  Devices::Sequential, short >,
   //Array< Quad< double >, Devices::Sequential, short >,
   Array< CustomScalar< int >, Devices::Sequential, int >,
   Array< int, Devices::Sequential, int >,
   Array< long, Devices::Sequential, int >,
   Array< double, Devices::Sequential, int >,
   //Array< Quad< float >,  Devices::Sequential, int >,
   //Array< Quad< double >, Devices::Sequential, int >,
   Array< CustomScalar< int >, Devices::Sequential, long >,
   Array< int, Devices::Sequential, long >,
   Array< long, Devices::Sequential, long >,
   Array< double, Devices::Sequential, long >,
   //Array< Quad< float >,  Devices::Sequential, long >,
   //Array< Quad< double >, Devices::Sequential, long >,

   Array< CustomScalar< int >, Devices::Host, short >,
   Array< int, Devices::Host, short >,
   Array< long, Devices::Host, short >,
   Array< double, Devices::Host, short >,
   //Array< Quad< float >,  Devices::Host, short >,
   //Array< Quad< double >, Devices::Host, short >,
   Array< CustomScalar< int >, Devices::Host, int >,
   Array< int, Devices::Host, int >,
   Array< long, Devices::Host, int >,
   Array< double, Devices::Host, int >,
   //Array< Quad< float >,  Devices::Host, int >,
   //Array< Quad< double >, Devices::Host, int >,
   Array< CustomScalar< int >, Devices::Host, long >,
   Array< int, Devices::Host, long >,
   Array< long, Devices::Host, long >,
   Array< double, Devices::Host, long >
//Array< Quad< float >,  Devices::Host, long >,
//Array< Quad< double >, Devices::Host, long >
#elif defined( __CUDACC__ )
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Cuda, short >,
   Array< int, Devices::Cuda, short >,
   Array< long, Devices::Cuda, short >,
   Array< double, Devices::Cuda, short >,
   //Array< Quad< float >,  Devices::Cuda, short >,
   //Array< Quad< double >, Devices::Cuda, short >,
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Cuda, int >,
   Array< int, Devices::Cuda, int >,
   Array< long, Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   //Array< Quad< float >,  Devices::Cuda, int >,
   //Array< Quad< double >, Devices::Cuda, int >,
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Cuda, long >,
   Array< int, Devices::Cuda, long >,
   Array< long, Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >
//Array< Quad< float >,  Devices::Cuda, long >,
//Array< Quad< double >, Devices::Cuda, long >
#elif defined( __HIP__ )
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Hip, short >,
   Array< int, Devices::Hip, short >,
   Array< long, Devices::Hip, short >,
   Array< double, Devices::Hip, short >,
   //Array< Quad< float >,  Devices::Hip, short >,
   //Array< Quad< double >, Devices::Hip, short >,
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Hip, int >,
   Array< int, Devices::Hip, int >,
   Array< long, Devices::Hip, int >,
   Array< double, Devices::Hip, int >,
   //Array< Quad< float >,  Devices::Hip, int >,
   //Array< Quad< double >, Devices::Hip, int >,
   // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< CustomScalar< int >, Devices::Hip, long >,
   Array< int, Devices::Hip, long >,
   Array< long, Devices::Hip, long >,
   Array< double, Devices::Hip, long >
//Array< Quad< float >,  Devices::Hip, long >,
//Array< Quad< double >, Devices::Hip, long >
#endif
   >;

TYPED_TEST_SUITE( ScanTest, ArrayTypes );

TYPED_TEST( ScanTest, inclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, inclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, inclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = ( i * ( i + 1 ) ) / 2;
   }

   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = ( i * ( i + 1 ) ) / 2;
   }

   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = ( i * ( i - 1 ) ) / 2;
   }

   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = ( i * ( i - 1 ) ) / 2;
   }

   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( ScanTest, multiplication )
{
   this->input_host.setSize( 10 );
   this->input_host.setValue( 2 );
   this->expected_host = this->input_host;

   // exclusive scan test
   int value = 1;
   for( int i = 0; i < this->expected_host.getSize(); i++ ) {
      this->expected_host[ i ] = value;
      value *= 2;
   }

   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->a.getSize(), 0, TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   for( int i = 0; i < this->expected_host.getSize(); i++ )
      this->expected_host[ i ] *= 2;

   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->a.getSize(), 0, TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, custom_begin_end )
{
   using IndexType = typename TestFixture::IndexType;

   const IndexType begin = 42;
   const IndexType end = this->size - begin;

   // exclusive scan test
   this->input_host.setValue( 1 );
   this->expected_host.setValue( 1 );
   for( int i = begin; i < end; i++ )
      this->expected_host[ i ] = i - begin;

   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, begin, end, begin );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   for( int i = begin; i < end; i++ )
      this->expected_host[ i ]++;

   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, begin, end, begin );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, empty_range )
{
   using IndexType = typename TestFixture::IndexType;

   this->input_host.setSize( 42 );
   this->input_host.setValue( 1 );
   this->expected_host = this->input_host;

   const IndexType begin = 2;
   const IndexType end = 1;

   // exclusive scan test
   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, begin, end, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, begin, end, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, vector_expression )
{
   this->a.setValue( 2 );
   this->b.setValue( 1 );

   // exclusive scan test
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   this->c.setValue( 0 );
   exclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->c );

   // inclusive scan test
   for( int i = 0; i < this->expected_host.getSize(); i++ )
      this->expected_host[ i ]++;

   this->c.setValue( 0 );
   inclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->c );
}

// Test that exercises the single-pass lookback scan kernel with non-trivial
// (non-constant) input data and the vectorized load/store path. The fixture
// sets maxGridSize=3 which forces the multi-grid (three-phase) path; this test
// restores maxGridSize so numberOfGrids==1 and the lookback kernel is selected.
// A linear sequence is used because the expected prefix sums are easy to
// verify and any wrong block-offset (the historical begin-threadOffset bug)
// produces a visible mismatch.
TYPED_TEST( ScanTest, lookback_kernel_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;
   using IndexType = typename TestFixture::IndexType;
   using DeviceType = typename TestFixture::DeviceType;

   // The fixture constrains maxGridSize to 3, which would route to the
   // three-phase path. Reset it so the lookback kernel is selected.
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
   }

   // Choose size large enough to span multiple blocks with the default
   // maxElementsInBlock (= blockSize * lookbackValuesPerThread). A prime
   // size ensures the last block is partial, exercising the identity-padding
   // path in Phase 1.
   const IndexType lookbackSize = 9377;

   this->input_host.setSize( lookbackSize );
   this->expected_host.setSize( lookbackSize );
   for( IndexType i = 0; i < lookbackSize; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = ( i * ( i + 1 ) ) / 2;
   }

   this->a.setSize( lookbackSize );
   this->b.setSize( lookbackSize );
   this->a = this->input_host;

   // Inclusive scan, array -> array (exercises vectorized load + store)
   inclusiveScan( this->a, this->b, 0, lookbackSize, 0, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->b;
      for( IndexType i = 0; i < lookbackSize; i++ )
         EXPECT_EQ( this->array_host[ i ], this->expected_host[ i ] ) << "inclusive array->array mismatch at i=" << i;
   }

   // In-place inclusive scan (exercises vectorized load + store on same array)
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, lookbackSize, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->a;
      for( IndexType i = 0; i < lookbackSize; i++ )
         EXPECT_EQ( this->array_host[ i ], this->expected_host[ i ] ) << "inclusive in-place mismatch at i=" << i;
   }

   // Exclusive scan, array -> array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, lookbackSize, 0, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->b;
      for( IndexType i = 0; i < lookbackSize; i++ )
         EXPECT_EQ( this->array_host[ i ], ( i * ( i - 1 ) ) / 2 ) << "exclusive array->array mismatch at i=" << i;
   }

   // In-place exclusive scan
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, lookbackSize, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->a;
      for( IndexType i = 0; i < lookbackSize; i++ )
         EXPECT_EQ( this->array_host[ i ], ( i * ( i - 1 ) ) / 2 ) << "exclusive in-place mismatch at i=" << i;
   }

   // Verify that the lookback kernel was actually used (gridsCount == 1)
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      const auto gridsCount = TNL::max(
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::gridsCount().load(),
         GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::gridsCount().load() );
      EXPECT_EQ( gridsCount, 1 );
   }
}

// Test that the lookback kernel handles an unaligned begin offset. The
// vectorized Vec4 load/store path requires the tile origin to be 16-byte
// aligned. With begin=3 and double (8 B), the byte offset is 24 (not a
// multiple of 16), which would crash ld.global.v4 if the kernel did not
// fall back to scalar loads. This test exercises the fallback.
TYPED_TEST( ScanTest, lookback_kernel_unaligned_begin )
{
   using ValueType = typename TestFixture::ValueType;
   using IndexType = typename TestFixture::IndexType;
   using DeviceType = typename TestFixture::DeviceType;

   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
   }

   // begin=3 breaks 16-byte alignment for double (3*8=24) and int (3*4=12).
   const IndexType size = 9377;
   const IndexType begin = 3;
   const IndexType scanSize = size - begin;

   this->input_host.setSize( size );
   this->expected_host.setSize( size );
   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );
   for( IndexType i = 0; i < scanSize; i++ ) {
      this->input_host[ begin + i ] = i;
      this->expected_host[ begin + i ] = ( i * ( i + 1 ) ) / 2;
   }

   this->a.setSize( size );
   this->b.setSize( size );
   this->a = this->input_host;

   inclusiveScan( this->a, this->b, begin, size, begin, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->b;
      for( IndexType i = 0; i < scanSize; i++ )
         EXPECT_EQ( this->array_host[ begin + i ], this->expected_host[ begin + i ] )
            << "inclusive unaligned begin mismatch at i=" << i;
   }

   this->a = this->input_host;
   inplaceInclusiveScan( this->a, begin, size, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->a;
      for( IndexType i = 0; i < scanSize; i++ )
         EXPECT_EQ( this->array_host[ begin + i ], this->expected_host[ begin + i ] )
            << "inclusive in-place unaligned begin mismatch at i=" << i;
   }
}

// Test that the lookback kernel handles views whose data do not start at a
// 16-byte boundary. Unlike lookback_kernel_unaligned_begin, the scan starts
// at index 0 of the view, so the alignment must be determined from the
// address and not from the index. The input and output views have different
// misalignments.
TYPED_TEST( ScanTest, lookback_kernel_unaligned_view )
{
   using ValueType = typename TestFixture::ValueType;
   using IndexType = typename TestFixture::IndexType;
   using DeviceType = typename TestFixture::DeviceType;

   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
   }

   const IndexType size = 9377;
   const IndexType inputOffset = 1;
   const IndexType outputOffset = 2;
   const IndexType scanSize = size - outputOffset;

   this->input_host.setSize( size );
   this->input_host.setValue( 0 );
   for( IndexType i = 0; i < scanSize; i++ )
      this->input_host[ inputOffset + i ] = i;

   this->a.setSize( size );
   this->b.setSize( size );
   this->a = this->input_host;
   this->b.setValue( 0 );

   auto inputView = this->a.getView( inputOffset, inputOffset + scanSize );
   auto outputView = this->b.getView( outputOffset, outputOffset + scanSize );

   inclusiveScan( inputView, outputView, 0, scanSize, 0, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->b;
      for( IndexType i = 0; i < scanSize; i++ )
         EXPECT_EQ( this->array_host[ outputOffset + i ], ( i * ( i + 1 ) ) / 2 ) << "inclusive unaligned view mismatch at i=" << i;
   }

   exclusiveScan( inputView, outputView, 0, scanSize, 0, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->b;
      for( IndexType i = 0; i < scanSize; i++ )
         EXPECT_EQ( this->array_host[ outputOffset + i ], ( i * ( i - 1 ) ) / 2 ) << "exclusive unaligned view mismatch at i=" << i;
   }

   inplaceInclusiveScan( inputView, 0, scanSize, std::plus<>{}, (ValueType) 0 );
   {
      this->array_host = this->a;
      for( IndexType i = 0; i < scanSize; i++ )
         EXPECT_EQ( this->array_host[ inputOffset + i ], ( i * ( i + 1 ) ) / 2 )
            << "inclusive in-place unaligned view mismatch at i=" << i;
   }
}

// Test that the scans performed concurrently by several host threads do not
// interfere. The lookback kernel stores the states of the tiles in an array
// which is cached by GPUScanKernelLauncher and reused by all scans (of the same
// launcher instantiation on the same device), so the launcher serializes these
// scans with a mutex. Each thread scans its own array with a different value,
// so mixing the states of different scans would change the results (or the
// kernel would wait forever for a state of another scan).
TYPED_TEST( ScanTest, lookback_kernel_concurrent_host_threads )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ValueType = typename TestFixture::ValueType;
   using IndexType = typename TestFixture::IndexType;
   using DeviceType = typename TestFixture::DeviceType;
   using HostArrayType = typename TestFixture::HostArrayType;

   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
      GPUScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
   }

   const IndexType size = 9377;
   const int threadsCount = 4;
   const int iterations = 50;
   std::vector< int > errors( threadsCount, 0 );

   auto work = [ & ]( int t )
   {
      ArrayType input( size );
      ArrayType output( size );
      HostArrayType result;
      input.setValue( ValueType( t + 1 ) );
      for( int iter = 0; iter < iterations; iter++ ) {
         output.setValue( ValueType( 0 ) );
         inclusiveScan( input, output, 0, size, 0, std::plus<>{}, (ValueType) 0 );
         result = output;
         for( IndexType i = 0; i < size; i++ )
            if( result[ i ] != ValueType( ( i + 1 ) * ( t + 1 ) ) )
               errors[ t ]++;
      }
   };

   std::vector< std::thread > threads;
   for( int t = 0; t < threadsCount; t++ )
      threads.emplace_back( work, t );
   for( auto& thread : threads )
      thread.join();

   for( int t = 0; t < threadsCount; t++ )
      EXPECT_EQ( errors[ t ], 0 ) << "wrong results in thread " << t;
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
// Test of the GPU scan with value types smaller than 4 bytes, which are
// shuffled as 4-byte words in the block scan. The results (with the overflows
// of the value type) are compared with the sequential scan on the host. The
// sizes cover the single-block kernel and the lookback kernel.
template< typename ValueType >
void
testSmallValueTypeScan()
{
   using DeviceType = Devices::GPU;
   using IndexType = int;
   using HostArrayType = Array< ValueType, Devices::Sequential, IndexType >;

   for( const IndexType size : { 100, 1000, 100003 } ) {
      HostArrayType input_host( size );
      for( IndexType i = 0; i < size; i++ )
         input_host[ i ] = ValueType( i % 7 );
      HostArrayType inclusive_host;
      HostArrayType exclusive_host;
      inclusive_host = exclusive_host = input_host;
      inplaceInclusiveScan( inclusive_host, 0, size, std::plus<>{}, ValueType( 0 ) );
      inplaceExclusiveScan( exclusive_host, 0, size, std::plus<>{}, ValueType( 0 ) );

      Array< ValueType, DeviceType, IndexType > input;
      Array< ValueType, DeviceType, IndexType > output( size );
      input = input_host;
      HostArrayType result;

      inclusiveScan( input, output, 0, size, 0, std::plus<>{}, ValueType( 0 ) );
      result = output;
      EXPECT_EQ( result, inclusive_host ) << "inclusive scan with size " << size;

      exclusiveScan( input, output, 0, size, 0, std::plus<>{}, ValueType( 0 ) );
      result = output;
      EXPECT_EQ( result, exclusive_host ) << "exclusive scan with size " << size;
   }
}

TEST( ScanSmallValueTypeTest, char )
{
   testSmallValueTypeScan< char >();
}

TEST( ScanSmallValueTypeTest, unsigned_char )
{
   testSmallValueTypeScan< unsigned char >();
}

TEST( ScanSmallValueTypeTest, short )
{
   testSmallValueTypeScan< short >();
}

TEST( ScanSmallValueTypeTest, unsigned_short )
{
   testSmallValueTypeScan< unsigned short >();
}

// Test of the lookback kernel with a 16-byte value type with 8-byte alignment,
// which uses the generic block scan (without the warp shuffle instructions).
// With an 8-byte offset of the data, no element starts at a 16-byte boundary,
// so the tiles must be accessed scalarly instead of via Vec4. The views are
// created over arrays of doubles, since the data of an array of the value type
// are always 16-byte aligned.
TEST( ScanStaticVectorTest, lookback_kernel_16_byte_value_type )
{
   using ValueType = StaticVector< 2, double >;
   using DeviceType = Devices::GPU;
   using IndexType = int;
   using ViewType = ArrayView< ValueType, DeviceType, IndexType >;
   using HostArrayType = Array< ValueType, Devices::Sequential, IndexType >;

   const IndexType size = 9377;
   Array< double, DeviceType, IndexType > inputStorage( 2 * size + 1 );
   Array< double, DeviceType, IndexType > outputStorage( 2 * size + 1 );

   HostArrayType input_host( size );
   HostArrayType inclusive_host( size );
   HostArrayType exclusive_host( size );
   for( IndexType i = 0; i < size; i++ ) {
      input_host[ i ] = ValueType{ double( i ), -double( i ) };
      inclusive_host[ i ] = ValueType{ i * ( i + 1 ) / 2.0, -i * ( i + 1 ) / 2.0 };
      exclusive_host[ i ] = ValueType{ i * ( i - 1 ) / 2.0, -i * ( i - 1 ) / 2.0 };
   }
   HostArrayType result;

   for( const IndexType offset : { 0, 1 } ) {
      ViewType input( reinterpret_cast< ValueType* >( inputStorage.getData() + offset ), size );
      ViewType output( reinterpret_cast< ValueType* >( outputStorage.getData() + offset ), size );

      input = input_host;
      inclusiveScan( input, output, 0, size, 0, std::plus<>{}, ValueType( 0 ) );
      result = output;
      EXPECT_EQ( result, inclusive_host ) << "inclusive scan with offset " << offset;

      exclusiveScan( input, output, 0, size, 0, std::plus<>{}, ValueType( 0 ) );
      result = output;
      EXPECT_EQ( result, exclusive_host ) << "exclusive scan with offset " << offset;

      inplaceInclusiveScan( input, 0, size, std::plus<>{}, ValueType( 0 ) );
      result = input;
      EXPECT_EQ( result, inclusive_host ) << "in-place inclusive scan with offset " << offset;
   }
}
#endif

#include "../main.h"
