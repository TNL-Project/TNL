#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

#include <gtest/gtest.h>

using namespace TNL;

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_1D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a;

   for( int size : { 1, 100, 10000, 1000000, 100000000 } ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel = [ = ] __cuda_callable__( int i ) mutable
      {
         view[ i ] = i;
      };
      Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 1D_cuda )
{
   test_1D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_2D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 2, int >;

   Array a;

   for( int size : { 1, 100, 10000, 1000000, 100000000 } ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ size, 1 }, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ 1, size }, kernel2 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 2D_cuda )
{
   test_2D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_3D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 3, int >;

   Array a;

   for( int size : { 1, 100, 10000, 1000000, 100000000 } ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ size, 1, 1 }, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, size, 1 }, kernel2 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel3 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.z() ] = i.z();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, size }, kernel3 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 3D_cuda )
{
   test_3D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_1D_above_32bit_range_cuda()
{
   // Regression test for a 32-bit overflow in the parallel-for kernel: in a
   // grid with more than 2^32 threads, the products of the block and thread
   // indices evaluated in 32 bits wrapped around, so the iterations above
   // 2^32 were never processed. Only the last `tail` iterations write into a
   // small array, so the test does not need large allocations.
   constexpr std::size_t size = ( 1UL << 32 ) + 1024;
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;  // == 2^32

   using Array = Containers::Array< int, Devices::Cuda, std::size_t >;
   using ArrayHost = Containers::Array< int, Devices::Host, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( std::size_t i ) mutable
   {
      if( i >= offset )
         view[ i - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel );

   ArrayHost ah;
   ah = a;
   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( ah[ j ], 1 ) << "The iteration i = 2^32 + " << j << " was not processed exactly once";
}

TEST( ParallelForTest, 1D_above_32bit_range_cuda )
{
   test_1D_above_32bit_range_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_2D_above_32bit_range_cuda()
{
   // Analogous to the 1D test, but the x-dimension of the 2D range exceeds
   // 2^32 threads, where the products of the block and grid dimensions
   // evaluated in 32 bits wrapped around. The size is a multiple of 2^31 above
   // 2^32, so the whole range still fits into a single grid. Only the last
   // `tail` iterations in the x-dimension write into a small array, so the
   // test does not need large allocations.
   constexpr std::size_t size = ( 1UL << 32 ) + ( 1UL << 31 );
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;

   using Array = Containers::Array< int, Devices::Cuda, std::size_t >;
   using ArrayHost = Containers::Array< int, Devices::Host, std::size_t >;
   using MultiIndex = Containers::StaticArray< 2, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      if( i.x() >= offset )
         view[ i.x() - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ size, 1 }, kernel );

   ArrayHost ah;
   ah = a;
   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( ah[ j ], 1 ) << "The 2D iteration with i.x() = " << offset + j << " was not processed exactly once";
}

TEST( ParallelForTest, 2D_above_32bit_range_cuda )
{
   test_2D_above_32bit_range_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_3D_above_32bit_range_cuda()
{
   // Analogous to the 2D test, but for a 3D range
   // (the x-dimension exceeds 2^32 threads, the others are trivial).
   constexpr std::size_t size = ( 1UL << 32 ) + ( 1UL << 31 );
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;

   using Array = Containers::Array< int, Devices::Cuda, std::size_t >;
   using ArrayHost = Containers::Array< int, Devices::Host, std::size_t >;
   using MultiIndex = Containers::StaticArray< 3, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      if( i.x() >= offset )
         view[ i.x() - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ size, 1, 1 }, kernel );

   ArrayHost ah;
   ah = a;
   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( ah[ j ], 1 ) << "The 3D iteration with i.x() = " << offset + j << " was not processed exactly once";
}

TEST( ParallelForTest, 3D_above_32bit_range_cuda )
{
   test_3D_above_32bit_range_cuda();
}

void
test_1D_empty_range_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( int i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Cuda >( 0, 0, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( 5, 3, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 1D_empty_range_cuda )
{
   test_1D_empty_range_cuda();
}

void
test_2D_empty_range_cuda()
{
   using MultiIndex = Containers::StaticArray< 2, int >;
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5 }, MultiIndex{ 1, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 5 }, MultiIndex{ 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5 }, MultiIndex{ 0, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 2D_empty_range_cuda )
{
   test_2D_empty_range_cuda();
}

void
test_3D_empty_range_cuda()
{
   using MultiIndex = Containers::StaticArray< 3, int >;
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 1, 1, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 1, 3, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 1, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5, 5 }, MultiIndex{ 10, 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0, 5 }, MultiIndex{ 3, 10, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 5, 0 }, MultiIndex{ 3, 3, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 5, 5 }, MultiIndex{ 3, 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 10, 0, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 10, 3, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 10, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 0, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 0, 3, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Cuda >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 0, 10, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 3D_empty_range_cuda )
{
   test_3D_empty_range_cuda();
}

#include "../main.h"
