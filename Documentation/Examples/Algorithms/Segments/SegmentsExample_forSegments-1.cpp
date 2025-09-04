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
   //! [config]
   using Device = typename Segments::DeviceType;

   /***
    * Create segments with given segments sizes.
    */
   Segments segments{ 1, 2, 3, 4, 5 };

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );
   //! [config]

   //! [traversing]
   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   using SegmentViewType = typename Segments::SegmentViewType;
   TNL::Algorithms::Segments::forAllSegments( segments,
                                              [ = ] __cuda_callable__( const SegmentViewType& segment ) mutable
                                              {
                                                 double sum( 0.0 );
                                                 for( auto element : segment )
                                                    if( element.localIndex() <= element.segmentIndex() ) {
                                                       sum += element.localIndex() + 1;
                                                       data_view[ element.globalIndex() ] = sum;
                                                    }
                                              } );
   //! [traversing]

   //! [printing]
   /***
    * Print the data managed by the segments.
    */
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   std::cout << TNL::Algorithms::Segments::print( segments, fetch ) << std::endl;
   //! [printing]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host:\n";
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int > >();

   std::cout << "Example of Ellpack segments on host:\n";
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int > >();

#ifdef __CUDACC__
   std::cout << "Example of CSR segments on CUDA GPU:\n";
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int > >();

   std::cout << "Example of Ellpack segments on CUDA GPU:\n";
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int > >();
#endif
   return EXIT_SUCCESS;
}
