#include "TNL/Functional.h"
#include <iostream>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/print.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Segments;

template< typename Device >
void
reduceAllExample()
{
   /***
    * This example shows how to perform a complete reduction over segments
    * with different operations for segment-level and final reduction.
    *
    * We will:
    * 1. Find maximum in each segment
    * 2. Sum up all the maximums
    */

   using IndexType = int;
   using ValueType = double;

   // Create segments with different sizes
   const IndexType segmentsCount = 4;
   Containers::Vector< IndexType, Device, IndexType > segmentsSizes{ 1, 2, 3, 4 };
   CSR< Device, IndexType > segments( segmentsSizes );

   // Initialize data: each element is segment_idx + local_idx
   Containers::Vector< ValueType, Device, IndexType > values( segments.getStorageSize(), -1 );
   auto valuesView = values.getView();
   auto segmentsSizesView = segmentsSizes.getView();
   segments.forAllElements(
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizesView[ segmentIdx ] )
            valuesView[ globalIdx ] = segmentIdx + localIdx;
      } );

   // Print the initial data
   std::cout << "Segments sizes: " << segmentsSizes << "\n";
   std::cout << Segments::print( segments,
                                 [ = ] __cuda_callable__( IndexType idx )
                                 {
                                    return valuesView[ idx ];
                                 } )
             << "\n";

   // Perform complete reduction:
   // 1. Find maximum in each segment
   // 2. Sum up all the maximums
   const ValueType result = reduceAll(
      segments,
      // Fetch function for segment-level reduction (gets element value)
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx )
      {
         if( localIdx < segmentsSizesView[ segmentIdx ] )
            return valuesView[ globalIdx ];
         return std::numeric_limits< ValueType >::lowest();
      },
      // Reduction operation for segments (maximum)
      TNL::Max{},
      // Fetch function for final reduction (identity - returns segment result as is)
      [] __cuda_callable__( const ValueType& segmentValue )
      {
         return segmentValue;
      },
      // Final reduction operation (sum)
      TNL::Plus{} );

   // Print the result
   std::cout << "Sum of maximums = " << result << "\n";
}

int
main( int argc, char** argv )
{
   std::cout << "Running example on Host:\n";
   reduceAllExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning example on CUDA device:\n";
   reduceAllExample< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
