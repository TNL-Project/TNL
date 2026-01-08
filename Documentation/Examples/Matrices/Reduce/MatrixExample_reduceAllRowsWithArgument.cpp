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
reduceAllRowsWithArgumentExample()
{
   /***
    * Create a 6x6 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 6, 6 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = ( rowIdx * 13 + columnIdx * 7 ) % 20;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Dense matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Find maximum value and its column index in each row.
    */
   TNL::Containers::Vector< double, Device > rowMaxValues( matrix.getRows() );
   TNL::Containers::Vector< int, Device > rowMaxColumns( matrix.getRows() );
   auto maxValues_view = rowMaxValues.getView();
   auto maxColumns_view = rowMaxColumns.getView();

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

   auto store = [ = ] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumns_view[ rowIdx ] = columnIdx;
   };

   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduction, store, std::numeric_limits< double >::lowest() );
   // You may also use TNL::MaxWithArg{} instead of defining your own reduction lambda.

   std::cout << "Row maxima values: " << rowMaxValues << std::endl;
   std::cout << "Column indices of maxima: " << rowMaxColumns << std::endl;

   /***
    * Find minimum value and its column index in each row.
    */
   TNL::Containers::Vector< double, Device > rowMinValues( matrix.getRows() );
   TNL::Containers::Vector< int, Device > rowMinColumns( matrix.getRows() );
   auto minValues_view = rowMinValues.getView();
   auto minColumns_view = rowMinColumns.getView();

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

   auto storeMin = [ = ] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, const double& value, bool emptyRow ) mutable
   {
      minValues_view[ rowIdx ] = value;
      if( ! emptyRow )
         minColumns_view[ rowIdx ] = columnIdx;
   };

   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reductionMin, storeMin, std::numeric_limits< double >::max() );
   // You may also use TNL::MinWithArg{} instead of defining your own reduction lambda.

   std::cout << "Row minima values: " << rowMinValues << std::endl;
   std::cout << "Column indices of minima: " << rowMinColumns << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceAllRowsWithArgumentExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceAllRowsWithArgumentExample< TNL::Devices::Cuda >();
#endif
}
