#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsIfExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Use forElementsIf to set elements only in even rows.
    */
   auto evenRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };

   auto setElements = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      if( columnIdx <= rowIdx )
         value = rowIdx + columnIdx;
   };

   TNL::Matrices::forElementsIf( denseMatrix, 0, denseMatrix.getRows(), evenRowCondition, setElements );
   std::cout << "Dense matrix with elements set only in even rows:" << std::endl;
   std::cout << denseMatrix << std::endl;

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 2, 3, 4, 5 }, 5 );

   /***
    * Use forElementsIf to set elements only in rows where rowIdx > 1.
    */
   auto rowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx > 1;
   };

   auto setSparseElements = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      if( rowIdx >= localIdx ) {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   TNL::Matrices::forElementsIf( sparseMatrix, 0, sparseMatrix.getRows(), rowCondition, setSparseElements );
   std::cout << "Sparse matrix with elements set only in rows where rowIdx > 1:" << std::endl;
   std::cout << sparseMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forElementsIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forElementsIfExample< TNL::Devices::Cuda >();
#endif
}
