#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Segments >
void
SegmentsExample()
{
   using Device = typename Segments::DeviceType;

   /***
    * Create segments with given segments sizes.
    */
   Segments segments{ 1, 2, 3, 4, 5 };

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );

   //! [traversing]
   /***
    * Insert data into particular segments with no check.
    */
   auto data_view = data.getView();
   TNL::Algorithms::Segments::forAllElementsIf(
      segments,
      //! [condition]
      [ = ] __cuda_callable__( int segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;  // Iterate only over even-indexed segments.
      },
      //! [condition]
      [ = ] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) mutable
      {
         if( localIdx <= segmentIdx )
            data_view[ globalIdx ] = segmentIdx;
      } );
   //! [traversing]

   /***
    * Print the data managed by the segments.
    */
   std::cout << "Array: " << data << std::endl;
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   std::cout << TNL::Algorithms::Segments::print( segments, fetch ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int > >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int > >();

#ifdef __CUDACC__
   std::cout << "Example of CSR segments on CUDA GPU: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int > >();

   std::cout << "Example of Ellpack segments on CUDA GPU: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int > >();
#endif
   return EXIT_SUCCESS;
}
