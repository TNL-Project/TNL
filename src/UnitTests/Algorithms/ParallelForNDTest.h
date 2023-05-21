#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelForND.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST

const int maxTestSize = 100000;

TEST( ParallelForNDTest, 5D_sequential )
{
   using Array = Containers::Array< int, Devices::Sequential >;
   using Coordinates = Containers::StaticVector< 5, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with coordinates kernels
      a = 0 ;
      auto coordinates_kernel1 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 0 ] ] = i[ 0 ];
      };
      Algorithms::ParallelForND< Devices::Sequential, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ size, 1, 1, 1, 1 }, coordinates_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 1 ] ] = i[ 1 ];
      };
      Algorithms::ParallelForND< Devices::Sequential, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, size, 1, 1, 1 }, coordinates_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel3 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 2 ] ] = i[ 2 ];
      };
      Algorithms::ParallelForND< Devices::Sequential, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, size, 1, 1 }, coordinates_kernel3 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel4 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 3 ] ] = i[ 3 ];
      };
      Algorithms::ParallelForND< Devices::Sequential, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, size, 1 }, coordinates_kernel4 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel5 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 4 ] ] = i[ 4 ];
      };
      Algorithms::ParallelForND< Devices::Sequential, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, 1, size }, coordinates_kernel5 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 1D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 1, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0;
      auto expanded_kernel = [=] (int i) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates( 0 ), Coordinates( size ), expanded_kernel );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0;
      auto coordinates_kernel = [=] ( Coordinates i) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates( 0 ), Coordinates( size ), coordinates_kernel );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 2D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 2, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0;
      auto expanded_kernel1 = [=] (int i, int j) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates{ 0, 0 }, Coordinates{ size, 1 }, expanded_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel2 = [=] (int i, int j) mutable
      {
         view[j] = j;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates{ 0, 0 }, Coordinates{ 1, size }, expanded_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0;
      auto coordinates_kernel1 = [=] ( const Coordinates& i ) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0 }, Coordinates{ size, 1 }, coordinates_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] ( const Coordinates& i ) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0 }, Coordinates{ 1, size }, coordinates_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 3D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 3, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0 ;
      auto expanded_kernel1 = [=] (int i, int j, int k) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ size, 1, 1 }, expanded_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel2 = [=] (int i, int j, int k) mutable
      {
         view[j] = j;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, size, 1 }, expanded_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel3 = [=] (int i, int j, int k) mutable
      {
         view[k] = k;
      };
      Algorithms::ParallelForND< Devices::Host, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, 1, size }, expanded_kernel3 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0 ;
      auto coordinates_kernel1 = [=] ( const Coordinates& i ) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ size, 1, 1 }, coordinates_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] ( const Coordinates& i ) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, size, 1 }, coordinates_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel3 = [=] ( const Coordinates& i ) mutable
      {
         view[i.z()] = i.z();
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, 1, size }, coordinates_kernel3 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 5D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 5, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      Array expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with coordinates kernels
      a = 0 ;
      auto coordinates_kernel1 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 0 ] ] = i[ 0 ];
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ size, 1, 1, 1, 1 }, coordinates_kernel1 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 1 ] ] = i[ 1 ];
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, size, 1, 1, 1 }, coordinates_kernel2 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel3 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 2 ] ] = i[ 2 ];
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, size, 1, 1 }, coordinates_kernel3 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel4 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 3 ] ] = i[ 3 ];
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, size, 1 }, coordinates_kernel4 );

      if( a != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( a[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel5 = [=] ( const Coordinates& i ) mutable
      {
         view[ i[ 4 ] ] = i[ 4 ];
      };
      Algorithms::ParallelForND< Devices::Host, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, 1, size }, coordinates_kernel5 );

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
   using Coordinates = Containers::StaticVector< 1, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0;
      auto expanded_kernel = [=] __cuda_callable__ (int i) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Cuda, true >::exec( Coordinates{ 0 }, Coordinates{ size }, expanded_kernel );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0;
      auto coordinates_kernel = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0 }, Coordinates{ size }, coordinates_kernel );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 1D_cuda )
{
   test_1D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_2D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 2, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0;
      auto expanded_kernel1 = [=] __cuda_callable__ (int i, int j) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Cuda, true >::exec( Coordinates{ 0, 0 }, Coordinates{ size, 1 }, expanded_kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel2 = [=] __cuda_callable__ (int i, int j) mutable
      {
         view[j] = j;
      };
      Algorithms::ParallelFor2D< Devices::Cuda >::exec( 0, 0, 1, size, expanded_kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0;
      auto coordinates_kernel1 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0 }, Coordinates{ size, 1 }, coordinates_kernel1 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0 }, Coordinates{ 1, size }, coordinates_kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 2D_cuda )
{
   test_2D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_3D_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 3, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 100)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with expanded kernels
      a = 0;
      auto expanded_kernel1 = [=] __cuda_callable__ (int i, int j, int k) mutable
      {
         view[i] = i;
      };
      Algorithms::ParallelForND< Devices::Cuda, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ size, 1, 1 }, expanded_kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel2 = [=] __cuda_callable__ (int i, int j, int k) mutable
      {
         view[j] = j;
      };
      Algorithms::ParallelForND< Devices::Cuda, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, size, 1}, expanded_kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto expanded_kernel3 = [=] __cuda_callable__ (int i, int j, int k) mutable
      {
         view[k] = k;
      };
      Algorithms::ParallelForND< Devices::Cuda, true >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, 1, size }, expanded_kernel3 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      ////
      // Tests with coordinates kernels
      a = 0;
      auto coordinates_kernel1 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.x()] = i.x();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ size, 1, 1 }, coordinates_kernel1 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.y()] = i.y();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, size, 1}, coordinates_kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel3 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[i.z()] = i.z();
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0 }, Coordinates{ 1, 1, size }, coordinates_kernel3 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], expected[i] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 3D_cuda )
{
   test_3D_cuda();
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_5d_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using Coordinates = Containers::StaticVector< 5, int >;
   Array a;
   for (int size = 1; size <= maxTestSize; size *= 10)
   {
      ArrayHost expected;
      expected.setSize( size );
      for (int i = 0; i < size; i++)
         expected[ i ] = i;

      a.setSize( size );
      auto view = a.getView();

      ////
      // Tests with coordinates kernels
      a = 0 ;
      auto coordinates_kernel1 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         //printf( "i[ 0 ] = %d \n", i[0] );
         view[ i[ 0 ] ] = i[ 0 ];
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ size, 1, 1, 1, 1 }, coordinates_kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel2 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[ i[ 1 ] ] = i[ 1 ];
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, size, 1, 1, 1 }, coordinates_kernel2 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel3 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[ i[ 2 ] ] = i[ 2 ];
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, size, 1, 1 }, coordinates_kernel3 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel4 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[ i[ 3 ] ] = i[ 3 ];
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, size, 1 }, coordinates_kernel4 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], i ) << "First index at which the result is wrong is i = " << i;
      }

      a = 0;
      auto coordinates_kernel5 = [=] __cuda_callable__ ( const Coordinates& i ) mutable
      {
         view[ i[ 4 ] ] = i[ 4 ];
      };
      Algorithms::ParallelForND< Devices::Cuda, false >::exec( Coordinates{ 0, 0, 0, 0, 0 }, Coordinates{ 1, 1, 1, 1, size }, coordinates_kernel5 );

      ah = a;
      if( ah != expected ) {
         for (int i = 0; i < size; i++)
            ASSERT_EQ( ah[i], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForNDTest, 5D_cuda )
{
   test_5d_cuda();
}

#endif
#endif

#include "../main.h"
