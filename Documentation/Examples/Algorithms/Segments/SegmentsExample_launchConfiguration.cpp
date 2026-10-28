#include <iostream>
#include <stdexcept>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
SegmentsExample()
{
   using namespace TNL::Algorithms::Segments;

   /***
    * Create CSR segments with given segments sizes.
    */
   CSR< Device, int > segments{ 1, 2, 3, 4, 5, 64, 300 };
   const int segmentsCount = segments.getSegmentCount();

   //! [traversal]
   /***
    * Set all elements to one using 8 threads per segment.
    */
   TNL::Containers::Vector< double, Device > data( segments.getStorageSize() );
   auto data_view = data.getView();
   const LaunchConfiguration uniform8( ThreadsToSegmentsMapping::Uniform, 8 );
   forAllElements(
      segments,
      [ = ] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) mutable
      {
         data_view[ globalIdx ] = 1.0;
      },
      uniform8 );
   //! [traversal]

   //! [reduction]
   /***
    * Compute the sums of elements in each segment using the Tiered mapping.
    */
   TNL::Containers::Vector< double, Device > sums( segmentsCount );
   auto sums_view = sums.getView();
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   auto store = [ = ] __cuda_callable__( int segmentIdx, const double& value ) mutable
   {
      sums_view[ segmentIdx ] = value;
   };
   const LaunchConfiguration tiered( ThreadsToSegmentsMapping::Tiered, 4 );
   reduceAllSegments( segments, fetch, TNL::Plus{}, store, tiered );
   std::cout << "Segment sizes: " << sums << '\n';
   //! [reduction]

   //! [list]
   /***
    * Run the same reduction with all launch configurations available for the segments.
    */
   for( const auto& [ launchConfig, tag ] : reductionLaunchConfigurations( segments ) ) {
      sums = 0.0;
      reduceAllSegments( segments, fetch, TNL::Plus{}, store, launchConfig );
      std::cout << "   " << tag << ": " << sums << '\n';
   }
   //! [list]

   //! [unsupported]
   /***
    * BiEllpack segments support only the Default mapping.
    */
   BiEllpack< Device, int > biEllpack{ 1, 2, 3, 4, 5, 64, 300 };
   try {
      forAllElements( biEllpack, [] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) {}, uniform8 );
   }
   catch( const std::invalid_argument& e ) {
      std::cout << "Unsupported launch configuration: " << e.what() << '\n';
   }
   //! [unsupported]
   std::cout << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of launch configurations on host:\n";
   SegmentsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Example of launch configurations on CUDA device:\n";
   SegmentsExample< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
