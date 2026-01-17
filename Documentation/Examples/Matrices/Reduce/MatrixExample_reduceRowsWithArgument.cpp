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
reduceRowsWithArgumentExample()
{
   /***
    * Create an 8x8 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 8, 8 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = ( rowIdx * 7 + columnIdx * 11 ) % 25;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Find argmax for rows 2-6 (range variant).
    */
   int rangeBegin = 2, rangeEnd = 7;
   int rangeSize = rangeEnd - rangeBegin;
   TNL::Containers::Vector< double, Device > rangeMaxValues( rangeSize );
   TNL::Containers::Vector< int, Device > rangeMaxColumns( rangeSize );
   auto rangeMaxValues_view = rangeMaxValues.getView();
   auto rangeMaxColumns_view = rangeMaxColumns.getView();

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

   auto storeRange =
      [ = ] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      rangeMaxValues_view[ rowIdx - rangeBegin ] = value;
      if( ! emptyRow )
         rangeMaxColumns_view[ rowIdx - rangeBegin ] = columnIdx;
   };

   TNL::Matrices::reduceRowsWithArgument(
      matrix, rangeBegin, rangeEnd, fetch, reduction, storeRange, std::numeric_limits< double >::lowest() );
   // You may also use TNL::MaxWithArg{} instead of defining your own reduction lambda.

   std::cout << "Maxima for rows 2-6:" << std::endl;
   std::cout << "  Values: " << rangeMaxValues << std::endl;
   std::cout << "  Columns: " << rangeMaxColumns << std::endl;

   /***
    * Find argmin for specific rows (array variant).
    */
   TNL::Containers::Array< int, Device > rowIndexes{ 1, 3, 5, 7 };
   TNL::Containers::Vector< double, Device > arrayMinValues( matrix.getRows() );
   TNL::Containers::Vector< int, Device > arrayMinColumns( matrix.getRows() );
   TNL::Containers::Vector< double, Device > compressedMinValues( rowIndexes.getSize() );
   TNL::Containers::Vector< int, Device > compressedMinColumns( rowIndexes.getSize() );
   auto arrayMinValues_view = arrayMinValues.getView();
   auto arrayMinColumns_view = arrayMinColumns.getView();
   auto compressedMinValues_view = compressedMinValues.getView();
   auto compressedMinColumns_view = compressedMinColumns.getView();

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

   auto storeArray = [ = ] __cuda_callable__(
                        int indexOfRowIdx, int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      arrayMinValues_view[ rowIdx ] = value;
      compressedMinValues_view[ indexOfRowIdx ] = value;
      if( ! emptyRow ) {
         arrayMinColumns_view[ rowIdx ] = columnIdx;
         compressedMinColumns_view[ indexOfRowIdx ] = columnIdx;
      }
   };

   TNL::Matrices::reduceRowsWithArgument(
      matrix, rowIndexes, fetch, reductionMin, storeArray, std::numeric_limits< double >::max() );
   // You may also use TNL::MinWithArg{} instead of defining your own reduction lambda.

   std::cout << "Minima for rows [1, 3, 5, 7]:" << std::endl;
   std::cout << "  Values: " << arrayMinValues << std::endl;
   std::cout << "  Columns: " << arrayMinColumns << std::endl;
   std::cout << "Compressed minima for rows [1, 3, 5, 7]: " << std::endl;
   std::cout << "  Values: " << compressedMinValues.getView( 0, rowIndexes.getSize() ) << std::endl;
   std::cout << "  Columns: " << compressedMinColumns.getView( 0, rowIndexes.getSize() ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceRowsWithArgumentExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceRowsWithArgumentExample< TNL::Devices::Cuda >();
#endif
}
