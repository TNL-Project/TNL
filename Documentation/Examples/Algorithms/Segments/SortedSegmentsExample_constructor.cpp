#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
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
   TNL::Containers::Vector< int, Device > segmentsSizes{ 1, 2, 3, 4, 5 };
   Segments segments( segmentsSizes );
   std::cout << "Segments sizes are: " << segments << std::endl;

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );

   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   TNL::Algorithms::Segments::forAllElements( segments,
                                              [ = ] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) mutable
                                              {
                                                 if( localIdx <= segmentIdx )
                                                    data_view[ globalIdx ] = segmentIdx;
                                              } );

   /***
    * Print the data managed by the segments.
    */
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   std::cout << TNL::Algorithms::Segments::print( segments, fetch ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   // ![sorted-segments-definition]
   std::cout << "Example of sorted CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int > > >();
   // ![sorted-segments-definition]

   std::cout << "Example of sorted Ellpack segments on host: " << std::endl;
   SegmentsExample<
      TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int > > >();

#ifdef __CUDACC__
   std::cout << "Example of sorted CSR segments on CUDA GPU: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int > > >();

   std::cout << "Example of sorted Ellpack segments on CUDA GPU: " << std::endl;
   SegmentsExample<
      TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int > > >();
#endif
   return EXIT_SUCCESS;
}
