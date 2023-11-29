#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Hip.h>

#include <gtest/gtest.h>

using namespace TNL;

TEST( DevicePointerHipTest, ConstructorTest )
{
   using TestType = TNL::Containers::StaticArray< 2, int >;
   TestType obj1;
   Pointers::DevicePointer< TestType, Devices::Hip > ptr1( obj1 );

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   TestType obj2( 1, 2 );
   Pointers::DevicePointer< TestType, Devices::Hip > ptr2( obj2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
}

TEST( DevicePointerHipTest, getDataTest )
{
   using TestType = TNL::Containers::StaticArray< 2, int >;
   TestType obj1( 1, 2 );
   Pointers::DevicePointer< TestType, Devices::Hip > ptr1( obj1 );

   Pointers::synchronizeSmartPointersOnDevice< Devices::Hip >();

   TestType aux;

   TNL_BACKEND_SAFE_CALL(
      hipMemcpy( (void*) &aux, &ptr1.getData< Devices::Hip >(), sizeof( TestType ), hipMemcpyDeviceToHost ) );

   ASSERT_EQ( aux[ 0 ], 1 );
   ASSERT_EQ( aux[ 1 ], 2 );
}

__global__
void
copyArrayKernel( const TNL::Containers::Array< int, Devices::Hip >* inArray, int* outArray )
{
   if( threadIdx.x < 2 ) {
      outArray[ threadIdx.x ] = ( *inArray )[ threadIdx.x ];
   }
}

__global__
void
copyArrayKernel2( const Pointers::DevicePointer< TNL::Containers::Array< int, Devices::Hip > > inArray, int* outArray )
{
   if( threadIdx.x < 2 ) {
      outArray[ threadIdx.x ] = ( *inArray )[ threadIdx.x ];
   }
}

TEST( DevicePointerHipTest, getDataArrayTest )
{
   using TestType = TNL::Containers::Array< int, Devices::Hip >;
   TestType obj;
   Pointers::DevicePointer< TestType > ptr( obj );

   ptr->setSize( 2 );
   ptr->setElement( 0, 1 );
   ptr->setElement( 1, 2 );

   Pointers::synchronizeSmartPointersOnDevice< Devices::Hip >();

   int *testArray_device, *testArray_host;
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &testArray_device, 2 * sizeof( int ) ) );
   // clang-format off
   copyArrayKernel<<< 1, 2 >>>( &ptr.getData< Devices::Hip >(), testArray_device );
   // clang-format on
   testArray_host = new int[ 2 ];
   TNL_BACKEND_SAFE_CALL( hipMemcpy( testArray_host, testArray_device, 2 * sizeof( int ), hipMemcpyDeviceToHost ) );

   ASSERT_EQ( testArray_host[ 0 ], 1 );
   ASSERT_EQ( testArray_host[ 1 ], 2 );

   // clang-format off
   copyArrayKernel2<<< 1, 2 >>>( ptr, testArray_device );
   // clang-format on
   TNL_BACKEND_SAFE_CALL( hipMemcpy( testArray_host, testArray_device, 2 * sizeof( int ), hipMemcpyDeviceToHost ) );

   ASSERT_EQ( testArray_host[ 0 ], 1 );
   ASSERT_EQ( testArray_host[ 1 ], 2 );

   delete[] testArray_host;
   TNL_BACKEND_SAFE_CALL( hipFree( testArray_device ) );
}

TEST( DevicePointerHipTest, nullptrAssignement )
{
   using TestType = Pointers::DevicePointer< double, Devices::Hip >;
   double o1 = 5;
   TestType p1( o1 ), p2( nullptr );

   // This should not crash
   p1 = p2;

   ASSERT_FALSE( p1 );
   ASSERT_FALSE( p2 );
}

TEST( DevicePointerHipTest, swap )
{
   using TestType = Pointers::DevicePointer< double, Devices::Hip >;
   double o1( 1 ), o2( 2 );
   TestType p1( o1 ), p2( o2 );

   p1.swap( p2 );

   ASSERT_EQ( *p1, 2 );
   ASSERT_EQ( *p2, 1 );
}

#include "../main.h"
