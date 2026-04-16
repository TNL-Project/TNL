#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsWithIndexesExample()
{
   /***
    * Create a 5x5 dense matrix and sparse matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 2, 3, 2, 1 }, 5 );

   /***
    * Create a vector with row indexes to process (rows 1, 2, and 4).
    */
   TNL::Containers::Vector< int, Device > rowIndexes{ 1, 2, 4 };

   /***
    * Use forElements with row indexes to set specific matrix rows.
    */
   auto setDenseElements = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx * 10 + columnIdx;
   };

   TNL::Matrices::forElements( denseMatrix, rowIndexes, setDenseElements );
   std::cout << "Dense matrix with selected rows set:\n";
   std::cout << denseMatrix << '\n';

   /***
    * Set sparse matrix elements for selected rows.
    */
   auto setSparseElements = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      columnIdx = localIdx;
      value = rowIdx + localIdx + 1;
   };

   TNL::Matrices::forElements( sparseMatrix, rowIndexes, setSparseElements );
   std::cout << "Sparse matrix with selected rows set:\n";
   std::cout << sparseMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   forElementsWithIndexesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   forElementsWithIndexesExample< TNL::Devices::Cuda >();
#endif
}
