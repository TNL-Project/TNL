#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
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
                                                    data_view[ globalIdx ] = segmentIdx + localIdx;
                                              } );

   /***
    * Print the data by the segments.
    */
   std::cout << "Values of elements after intial setup: " << std::endl;
   auto fetch = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   std::cout << TNL::Algorithms::Segments::print( segments, fetch ) << std::endl;

   //! [reduction]
   /***
    * Find the maximum element in each segment.
    */
   TNL::Containers::Vector< double, Device > sums( size ), positions( size );
   auto sums_view = sums.getView();
   auto positions_view = positions.getView();
   auto fetch_full = [ = ] __cuda_callable__( int segmentIdx, int localIdx, int globalIdx ) -> double
   {
      if( localIdx <= segmentIdx )
         return data_view[ globalIdx ];
      else
         return 0.0;
   };
   auto fetch_brief = [ = ] __cuda_callable__( int globalIdx ) -> double
   {
      return data_view[ globalIdx ];
   };
   auto keep = [ = ] __cuda_callable__( int segmentIdx, int localIdx, const double& value ) mutable
   {
      sums_view[ segmentIdx ] = value;
      positions_view[ segmentIdx ] = localIdx;
   };

   TNL::Algorithms::Segments::reduceAllSegmentsWithArgument( segments, fetch_full, TNL::MaxWithArg{}, keep );
   std::cout << "The sums with full fetch form are: " << sums << std::endl;
   std::cout << "The positions of the largest elements are: " << positions << std::endl;

   sums = 0;
   positions = 0;
   TNL::Algorithms::Segments::reduceAllSegmentsWithArgument( segments, fetch_brief, TNL::MaxWithArg{}, keep );
   std::cout << "The sums with brief fetch form are: " << sums << std::endl;
   std::cout << "The positions of the largest elements are: " << positions << std::endl;
   //! [reduction]
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
