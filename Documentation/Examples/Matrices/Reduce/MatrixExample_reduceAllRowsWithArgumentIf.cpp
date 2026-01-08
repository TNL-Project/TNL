#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
reduceAllRowsWithArgumentIfExample()
{
   /***
    * Create a 6x6 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 6, 6 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = ( rowIdx * 7 + columnIdx * 11 ) % 20;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Dense matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Find argmax only for even-indexed rows.
    */
   TNL::Containers::Vector< double, Device > maxValues( matrix.getRows() );
   TNL::Containers::Vector< int, Device > maxColumns( matrix.getRows() );
   TNL::Containers::Vector< double, Device > compressedMaxValues( matrix.getRows() );
   TNL::Containers::Vector< int, Device > compressedMaxColumns( matrix.getRows() );

   auto maxValues_view = maxValues.getView();
   auto maxColumns_view = maxColumns.getView();
   auto compressedMaxValues_view = compressedMaxValues.getView();
   auto compressedMaxColumns_view = compressedMaxColumns.getView();

   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return value;
   };

   auto reduction = [] __cuda_callable__( double& a, const double& b, int& aIdx, const int& bIdx )
   {
      if( a < b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };

   auto rowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;  // Only even row indices
   };

   auto store =
      [ = ] __cuda_callable__( int indexOfRowIdx, int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      compressedMaxValues_view[ indexOfRowIdx ] = value;
      if( ! emptyRow ) {
         maxColumns_view[ rowIdx ] = columnIdx;
         compressedMaxColumns_view[ indexOfRowIdx ] = columnIdx;
      }
   };

   // Initialize with -1 to see which rows were processed
   maxValues.setValue( -1.0 );
   maxColumns.setValue( -1 );

   auto evenRowsCount = TNL::Matrices::reduceAllRowsWithArgumentIf(
      matrix, rowCondition, fetch, reduction, store, std::numeric_limits< double >::lowest() );
   // You may also use TNL::MaxWithArg{} instead of defining your own reduction lambda.

   std::cout << "Argmax for even-indexed rows:" << std::endl;
   std::cout << "  Max values (odd indices show -1): " << maxValues << std::endl;
   std::cout << "  Column indices: " << maxColumns << std::endl;
   std::cout << "Compressed argmax for even-indexed rows:" << std::endl;
   std::cout << "  Max values: " << compressedMaxValues.getView( 0, evenRowsCount ) << std::endl;
   std::cout << "  Column indices: " << compressedMaxColumns.getView( 0, evenRowsCount ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceAllRowsWithArgumentIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceAllRowsWithArgumentIfExample< TNL::Devices::Cuda >();
#endif
}
