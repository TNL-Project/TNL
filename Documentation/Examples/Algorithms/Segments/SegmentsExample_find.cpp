#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/find.h>
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
   const int size = 5;
   Segments segments{ 1, 2, 3, 4, 5 };

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
                                                    data_view[ globalIdx ] = int( localIdx + segmentIdx ) / 2;
                                              } );

   /***
    * Print the data by the segments.
    */
   std::cout << "Values of elements after initial setup: " << std::endl;
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   std::cout << TNL::Algorithms::Segments::print( segments, fetch ) << std::endl;

   //! [find]
   TNL::Containers::Vector< bool, Device, int > found( size, false );
   TNL::Containers::Vector< int, Device, int > positions( size, -1 );

   auto found_view = found.getView();
   auto positions_view = positions.getView();
   auto condition = [ = ] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) -> bool
   {
      return data_view[ globalIdx ] == 2;
   };
   auto keep = [ = ] __cuda_callable__( const int segmentIdx, const int localIdx, bool found ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInAllSegments( segments, condition, keep );

   std::cout << "Found array:     " << found << std::endl;
   std::cout << "Positions array: " << positions << std::endl;
   //! [find]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int > >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int > >();

#ifdef __CUDACC__
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int > >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int > >();
#endif
   return EXIT_SUCCESS;
}
