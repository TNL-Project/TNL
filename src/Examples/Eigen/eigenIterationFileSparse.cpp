#include "TNL/Backend/DeviceInfo.h"
#include "TNL/Devices/Sequential.h"
#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIterationShift.h>
#include <TNL/Algorithms/fillRandom.h>
#include <TNL/Matrices/MatrixReader.h>
#include <ostream>

template< typename Device >
void
smallEigenExample()
{
   using SparseMatrix = TNL::Matrices::SparseMatrix< double, Device >;
   SparseMatrix eigenMatrix;
   std::cout << "Reading sparse matrix from MTX file bcsstk01.mtx ... ";
   TNL::Matrices::MatrixReader< SparseMatrix >::readMtx( "/home/salabmar/TNL_Projects/tnl/.test_matrices/bcsstk01.mtx",
                                                         eigenMatrix );
   std::cout << "OK " << std::endl;
   int size = eigenMatrix.getColumns();
   for(int i = 0; i < 1; i++)
   {
   auto start = std::chrono::high_resolution_clock::now();
   auto [eigenvalue, eigenvector, iterations] =
      TNL::Matrices::Eigen::powerIterationShiftTuple< double, Device >( eigenMatrix, 1e-8, 0);
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration< double > elapsed = end - start;
   std::cout << "Elapsed time powerIteration: " << elapsed.count() << " seconds" << std::endl;
   std::cout << "Eigenvalue: " << eigenvalue << std::endl;
   std::cout << "Iterations: " << iterations << std::endl;
   TNL::Containers::Vector< double, Device > matrixEigenvector( size );
   eigenMatrix.vectorProduct( eigenvector, matrixEigenvector );
   TNL::Containers::Vector< double, Device > eigenProduct( size );
   eigenProduct.forAllElements(
      [ eigenvalue, eigenvector ] __cuda_callable__( int i, double& value )
      {
         value = eigenvector[ i ] * eigenvalue;
      } );
   double error = TNL::maxNorm( eigenProduct - matrixEigenvector );
   std::cout << "Max error = " << error << std::endl;
   }
}

int
main( int argc, char* argv[] )
{
   // std::cout << "Example of eigenvalue computation by power iteration method on CPU:" << std::endl;
   // smallEigenExample< TNL::Devices::Host >();

   // std::cout << "Example of eigenvalue computation by power iteration method on sequential device:" << std::endl;
   // smallEigenExample< TNL::Devices::Sequential >();

#ifdef __CUDACC__
   std::cout << "Example of eigenvalue computation by power iteration method on CUDA GPU:" << std::endl;
   smallEigenExample< TNL::Devices::Cuda >();
#endif
}
