#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST
TEST( ParallelForTest, 1D_host )
{
   using Array = Containers::Array< int, Devices::Host >;

   Array a;

   for (int size = 1; size <= 100000; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel = [=] (int i) mutable
      {
         view[i] = i;
      };
      Algorithms::parallelFor< Devices::Host >( 0, size, kernel );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 2D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 2, int >;

   Array a;

   for (int size = 1; size <= 100000; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [=] (const MultiIndex& i) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{0, 0}, MultiIndex{size, 1}, kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [=] (const MultiIndex& i) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{0, 0}, MultiIndex{1, size}, kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 3D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 3, int >;

   Array a;

   for (int size = 1; size <= 100000; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [=] (const MultiIndex& i) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{0, 0, 0}, MultiIndex{size, 1, 1}, kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [=] (const MultiIndex& i) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{0, 0, 0}, MultiIndex{1, size, 1}, kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel3 = [=] (const MultiIndex& i) mutable
      {
         view[i.z()] = i.z();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{0, 0, 0}, MultiIndex{1, 1, size}, kernel3 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

#ifdef __CUDACC__
// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_1D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a;

   for (int size = 1; size <= 100000000; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel = [=] __cuda_callable__ (int i) mutable
      {
         view[i] = i;
      };
      Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 1D_cuda )
{
   test_1D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_2D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 2, int >;

   Array a;

   for (int size = 1; size <= 100000000; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [=] __cuda_callable__ (const MultiIndex& i) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{0, 0}, MultiIndex{size, 1}, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [=] __cuda_callable__ (const MultiIndex& i) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{0, 0}, MultiIndex{1, size}, kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 2D_cuda )
{
   test_2D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_3D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 3, int >;

   Array a;

   for (int size = 1; size <= 100000000; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [=] __cuda_callable__ (const MultiIndex& i) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{0, 0, 0}, MultiIndex{size, 1, 1}, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [=] __cuda_callable__ (const MultiIndex& i) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{0, 0, 0}, MultiIndex{1, size, 1}, kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel3 = [=] __cuda_callable__ (const MultiIndex& i) mutable
      {
         view[i.z()] = i.z();
      };
      Algorithms::parallelFor< Devices::Cuda >( MultiIndex{0, 0, 0}, MultiIndex{1, 1, size}, kernel3 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 3D_cuda )
{
   test_3D_cuda();
}
#endif
#endif

#include "../main.h"
