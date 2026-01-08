#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
reduceRowsWithArgumentIfExample()
{
   /***
    * Create a 10x10 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 10, 10 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = ( rowIdx * 13 + columnIdx * 7 ) % 30;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Find argmax for rows 3-8, but only for even-indexed rows (range + condition).
    */
   int rangeBegin = 3, rangeEnd = 9;
   int rangeSize = rangeEnd - rangeBegin;
   TNL::Containers::Vector< double, Device > rangeMaxValues( rangeSize );
   TNL::Containers::Vector< int, Device > rangeMaxColumns( rangeSize );
   TNL::Containers::Vector< double, Device > compressedMaxValues( rangeSize );
   TNL::Containers::Vector< int, Device > compressedMaxColumns( rangeSize );
   auto rangeMaxValues_view = rangeMaxValues.getView();
   auto rangeMaxColumns_view = rangeMaxColumns.getView();
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

   auto evenRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };

   auto storeRange = [ = ] __cuda_callable__(
                        int indexOfRowIdx, int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      rangeMaxValues_view[ rowIdx - rangeBegin ] = value;
      compressedMaxValues_view[ indexOfRowIdx ] = value;
      if( ! emptyRow ) {
         rangeMaxColumns_view[ rowIdx - rangeBegin ] = columnIdx;
         compressedMaxColumns_view[ indexOfRowIdx ] = columnIdx;
      }
   };

   rangeMaxValues.setValue( -1.0 );
   rangeMaxColumns.setValue( -1 );

   auto evenRowsCount = TNL::Matrices::reduceRowsWithArgumentIf(
      matrix, rangeBegin, rangeEnd, evenRowCondition, fetch, reduction, storeRange, std::numeric_limits< double >::lowest() );
   // You may also use TNL::MaxWithArg{} instead of defining your own reduction lambda.

   std::cout << "Argmax for rows 3-8 (only even indices):" << std::endl;
   std::cout << "  Max values: " << rangeMaxValues << std::endl;
   std::cout << "  Column indices: " << rangeMaxColumns << std::endl;
   std::cout << "  Compressed max values: " << compressedMaxValues.getView( 0, evenRowsCount ) << std::endl;
   std::cout << "  Compressed column indices: " << compressedMaxColumns.getView( 0, evenRowsCount ) << std::endl;

   /***
    * Find argmin for rows 5-9, but only for odd-indexed rows.
    */
   TNL::Containers::Vector< double, Device > oddMinValues( 5 );
   TNL::Containers::Vector< int, Device > oddMinColumns( 5 );
   TNL::Containers::Vector< double, Device > compressedOddMinValues( 5 );
   TNL::Containers::Vector< int, Device > compressedOddMinColumns( 5 );
   auto oddMinValues_view = oddMinValues.getView();
   auto oddMinColumns_view = oddMinColumns.getView();
   auto compressedOddMinValues_view = compressedOddMinValues.getView();
   auto compressedOddMinColumns_view = compressedOddMinColumns.getView();

   auto oddRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 1;
   };

   auto reductionMin = [] __cuda_callable__( double& a, const double& b, int& aIdx, const int& bIdx )
   {
      if( a > b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };

   auto storeOddMin =
      [ = ] __cuda_callable__(
         int indexOfRowIdx, int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      oddMinValues_view[ rowIdx - 5 ] = value;
      compressedOddMinValues_view[ indexOfRowIdx ] = value;
      if( ! emptyRow ) {
         oddMinColumns_view[ rowIdx - 5 ] = columnIdx;
         compressedOddMinColumns_view[ indexOfRowIdx ] = columnIdx;
      }
   };

   oddMinValues.setValue( -1.0 );
   oddMinColumns.setValue( -1 );

   auto oddRowsCount = TNL::Matrices::reduceRowsWithArgumentIf(
      matrix, 5, 10, oddRowCondition, fetch, reductionMin, storeOddMin, std::numeric_limits< double >::max() );
   // You may also use TNL::MinWithArg{} instead of defining your own reduction lambda.

   std::cout << "Argmin for rows 5-9 (only odd indices):" << std::endl;
   std::cout << "  Min values: " << oddMinValues << std::endl;
   std::cout << "  Column indices: " << oddMinColumns << std::endl;
   std::cout << "  Compressed min values: " << compressedOddMinValues.getView( 0, oddRowsCount ) << std::endl;
   std::cout << "  Compressed column indices: " << compressedOddMinColumns.getView( 0, oddRowsCount ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceRowsWithArgumentIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceRowsWithArgumentIfExample< TNL::Devices::Cuda >();
#endif
}
