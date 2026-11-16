#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/UniquePointer.h>

using namespace TNL;

using ArrayGPU = Containers::Array< int, Devices::GPU >;

#if defined( __CUDACC__ ) || defined( __HIP__ )
__global__
void
printArray( const ArrayGPU* ptr )
{
   printf( "Array size is: %d\n", ptr->getSize() );
   for( int i = 0; i < ptr->getSize(); i++ )
      printf( "a[ %d ] = %d \n", i, ( *ptr )[ i ] );
}
#endif

int
main( int argc, char* argv[] )
{
   /***
    * Create an array and print its elements in GPU kernel
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Pointers::UniquePointer< ArrayGPU > array_ptr( 10 );
   array_ptr.modifyData< Devices::Host >() = 1;
   Pointers::synchronizeSmartPointersOnDevice< Devices::GPU >();
   printArray<<< 1, 1 >>>( &array_ptr.getData< Devices::GPU >() );

   /***
    * Resize the array and print it again
    */
   array_ptr.modifyData< Devices::Host >().setSize( 5 );
   array_ptr.modifyData< Devices::Host >() = 2;
   Pointers::synchronizeSmartPointersOnDevice< Devices::GPU >();
   printArray<<< 1, 1 >>>( &array_ptr.getData< Devices::GPU >() );
#endif
   return EXIT_SUCCESS;
}
