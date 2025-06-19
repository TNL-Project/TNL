#include <iostream>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/sort.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>

template< typename Device, typename Value = double, typename Index = int >
void
sortExample()
{
   // Create segments with different sizes
   TNL::Containers::Vector< Index, Device > segmentsSizes{ 1, 2, 3, 4, 5 };

   TNL::Algorithms::Segments::CSR< Device, Index > segments( segmentsSizes );

   // Create data to be sorted within segments
   TNL::Containers::Vector< Value, Device, Index > data( segments.getStorageSize(), -1 );

   // Initialize data
   auto data_view = data.getView();
   auto segmentsSizesView = segmentsSizes.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( Index segmentIdx, Index localIdx, Index globalIdx ) mutable
      {
         if( localIdx < segmentsSizesView[ segmentIdx ] )
            data_view[ globalIdx ] = segmentsSizesView[ segmentIdx ] - localIdx;
      } );

   // Print original data
   std::cout << "Original data in segments:" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> int
                                                  {
                                                     return data_view[ globalIdx ];
                                                  } )
             << std::endl;

   // Sort each segment
   auto fetch = [ = ] __cuda_callable__( Index segmentIdx, Index localIdx, Index globalIdx ) -> int
   {
      return data_view[ globalIdx ] != -1 ? data_view[ globalIdx ] : std::numeric_limits< int >::max();
   };
   auto compare = [] __cuda_callable__( const Value& a, const Value& b ) -> bool
   {
      return a <= b;
   };
   auto swap = [ = ] __cuda_callable__( Index globalIdx1, Index globalIdx2 ) mutable
   {
      TNL::swap( data_view[ globalIdx1 ], data_view[ globalIdx2 ] );
   };

   // Sort all segments
   TNL::Algorithms::Segments::sortAllSegments( segments, fetch, compare, swap );

   // Print sorted data
   std::cout << "\nSorted data in segments (ascending order):" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> int
                                                  {
                                                     return data_view[ globalIdx ];
                                                  } )
             << std::endl;

   // Sort only specific segments using segmentIndexes
   TNL::Containers::Vector< Index, Device > segmentIndexes{ 1, 3 };
   std::cout << "\nSorting only segments 1 and 3 in descending order:" << std::endl;

   auto compareDesc = [] __cuda_callable__( Index a, Index b ) -> bool
   {
      return a >= b;
   };

   TNL::Algorithms::Segments::sortSegments( segments, segmentIndexes, fetch, compareDesc, swap );

   // Print result
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> int
                                                  {
                                                     return data_view[ globalIdx ];
                                                  } )
             << std::endl;

   // Sort segments conditionally (only even-indexed segments)
   std::cout << "\nSorting even-indexed segments in descending order:" << std::endl;
   auto condition = [] __cuda_callable__( Index segmentIdx ) -> bool
   {
      return segmentIdx % 2 == 0;
   };

   TNL::Algorithms::Segments::sortAllSegmentsIf( segments, condition, fetch, compareDesc, swap );

   // Print result
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> int
                                                  {
                                                     return data_view[ globalIdx ];
                                                  } )
             << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running example on Host:\n";
   sortExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running example on Cuda:\n";
   sortExample< TNL::Devices::Cuda >();
#endif

   return 0;
}
