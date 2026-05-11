#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/SequentialFor.h>
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
   const int size( 5 );
   Segments segments{ 1, 2, 3, 4, 5 };
   auto view = segments.getView();

   /***
    * Print the elements mapping using segment view.
    */
   std::cout << "Mapping of local indexes to global indexes:\n";

   auto f = [ = ] __cuda_callable__( int segmentIdx )
   {
      printf( "Segment idx. %d: ", segmentIdx );  // printf works even in GPU kernels
      auto segment = view.getSegmentView( segmentIdx );
      for( auto element : segment )
         printf( "%d -> %d \t", element.localIndex(), element.globalIndex() );
      printf( "\n" );
   };
   TNL::Algorithms::SequentialFor< Device >::exec( 0, size, f );
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
