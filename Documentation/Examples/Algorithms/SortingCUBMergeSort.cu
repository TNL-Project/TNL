#include <iostream>
#include <TNL/Algorithms/Sorting/CUBMergeSort.h>
#include <TNL/Algorithms/fillRandom.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Cuda.h>

int
main( int argc, char* argv[] )
{
   const int size = 20;

   TNL::Containers::Array< int, TNL::Devices::Cuda > arr( size );

   /***
    * Fill the array with random integers.
    */
   TNL::Algorithms::fillRandom< TNL::Devices::Cuda >( arr.getData(), size, 0, 2 * size );

   std::cout << "Random array: " << arr << "\n";

   /***
    * Sort in ascending order.
    */
   TNL::Algorithms::Sorting::CUBMergeSort::sort( arr );

   std::cout << "Array sorted in ascending order: " << arr << "\n";

   return EXIT_SUCCESS;
}
