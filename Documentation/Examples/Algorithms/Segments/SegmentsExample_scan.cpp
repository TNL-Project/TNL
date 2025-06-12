#include <iostream>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/scan.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>

template< typename Device, typename Value = double, typename Index = int >
void
scanExample()
{
   // Create segments with different sizes
   TNL::Containers::Vector< Index, Device > segmentsSizes{ 1, 2, 3, 4, 5 };
   auto segmentsSizesView = segmentsSizes.getConstView();
   TNL::Algorithms::Segments::CSR< Device, Index > segments( segmentsSizes );

   // Create data to be scanned within segments
   TNL::Containers::Vector< Value, Device, Index > data( segments.getStorageSize() );
   TNL::Containers::Vector< Value, Device, Index > inclusive_result( segments.getStorageSize() );
   TNL::Containers::Vector< Value, Device, Index > exclusive_result( segments.getStorageSize() );
   auto inclusive_result_view = inclusive_result.getView();
   auto exclusive_result_view = exclusive_result.getView();

   // Initialize data with segment index + 1
   auto dataView = data.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( Index segmentIdx, Index localIdx, Index globalIdx ) mutable
      {
         dataView[ globalIdx ] = segmentIdx + 1;
      } );

   // Print original data
   std::cout << "Original data in segments:" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> Value
                                                  {
                                                     return dataView[ globalIdx ];
                                                  } )
             << std::endl;

   // Define fetch, reduce and write functions
   auto fetch = [ = ] __cuda_callable__( Index segmentIdx, Index localIdx, Index globalIdx ) -> Value
   {
      if( localIdx < segmentsSizesView[ segmentIdx ] )
         return dataView[ globalIdx ];
      else
         return 0;  // Return 0 for padding elements
   };
   auto write_inclusive = [ = ] __cuda_callable__( Index globalIdx, Value value ) mutable
   {
      inclusive_result_view[ globalIdx ] = value;
   };

   auto write_exclusive = [ = ] __cuda_callable__( Index globalIdx, Value value ) mutable
   {
      exclusive_result_view[ globalIdx ] = value;
   };

   // Perform inclusive scan
   TNL::Algorithms::Segments::inclusiveScanAllSegments( segments, fetch, TNL::Plus{}, write_inclusive );

   // Perform exclusive scan
   TNL::Algorithms::Segments::exclusiveScanAllSegments( segments, fetch, TNL::Plus{}, write_exclusive );

   // Print results
   std::cout << "\nInclusive scan results:" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> Value
                                                  {
                                                     return inclusive_result[ globalIdx ];
                                                  } )
             << std::endl;

   std::cout << "\nExclusive scan results:" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> Value
                                                  {
                                                     return exclusive_result[ globalIdx ];
                                                  } )
             << std::endl;

   // Example of scanning only specific segments
   TNL::Containers::Vector< Index, Device > segmentIndexes{ 1, 3 };  // Scan only segments 1 and 3

   auto write_partial = [ = ] __cuda_callable__( Index globalIdx, Value value ) mutable
   {
      dataView[ globalIdx ] = value;  // All segment scan algorithms may work even as inplace scan
   };

   // Perform inclusive scan on selected segments
   TNL::Algorithms::Segments::inclusiveScanSegments( segments, segmentIndexes, fetch, TNL::Plus{}, write_partial );

   std::cout << "\nPartial inclusive inplace scan results (only segments 1 and 3):" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> Value
                                                  {
                                                     return dataView[ globalIdx ];
                                                  } )
             << std::endl;

   // Scanning the rest of segments using condition
   auto condition = [ = ] __cuda_callable__( Index segmentIdx ) mutable -> bool
   {
      return segmentIdx % 2 == 0;  // Only scan even segments
   };

   // Perform inclusive scan on selected segments
   TNL::Algorithms::Segments::inclusiveScanAllSegmentsIf( segments, condition, fetch, TNL::Plus{}, write_partial );

   std::cout << "\nPartial inclusive inplace scan results (only even segments):" << std::endl;
   std::cout << TNL::Algorithms::Segments::print( segments,
                                                  [ = ] __cuda_callable__( Index globalIdx ) -> Value
                                                  {
                                                     return dataView[ globalIdx ];
                                                  } )
             << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running example on Host:\n";
   scanExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning example on Cuda:\n";
   scanExample< TNL::Devices::Cuda >();
#endif

   return 0;
}
