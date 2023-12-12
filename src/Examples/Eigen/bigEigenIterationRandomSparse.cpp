#include "TNL/Algorithms/Segments/ElementsOrganization.h"
#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Cuda/CudaCallable.h"
#include "TNL/Devices/Sequential.h"
#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Algorithms/fillRandom.h>
#include <ostream>

/*
Nastavit podle organizace paměti matice.

Sparse Matrix
1. setRowCapacities
2. getValues a pak fillRandom
3. getColumnIndexes();
*/

template< typename Device >
void
bigEigenExample()
{
   int size = 10000000;
   for( int i = 0; i < 1; i++ ) {
      TNL::Matrices::SparseMatrix< double, Device > randomMatrix( size, size );
      TNL::Containers::Vector< int, Device > rowCapacities( size );
      auto start = std::chrono::high_resolution_clock::now();
      TNL::Algorithms::fillRandom< Device >( rowCapacities.getData(), size, 1, 10 );
      randomMatrix.setRowCapacities( rowCapacities );
      TNL::Algorithms::fillRandom< Device >(
         randomMatrix.getColumnIndexes().getData(), randomMatrix.getColumnIndexes().getSize(), 0, size - 1 );
      TNL::Algorithms::fillRandom< Device >(
         randomMatrix.getValues().getData(), randomMatrix.getValues().getSize(), (double) 0, (double) 1 );
      //std::cout << randomMatrix << std::endl;
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > elapsed = end - start;
      std::cout << "Elapsed time fillRandom: " << elapsed.count() << " seconds" << std::endl;

      std::pair< double, TNL::Containers::Vector< double, Device > > eigen =
         TNL::Matrices::Eigen::powerIteration< double, Device >( randomMatrix, 1e-7 );
      start = std::chrono::high_resolution_clock::now();
      elapsed = start - end;
      std::cout << "Elapsed time powerIteration: " << elapsed.count() << " seconds" << std::endl;
      TNL::Containers::Vector< double, Device > matrixEigenvector( size );
      randomMatrix.vectorProduct( eigen.second, matrixEigenvector );
      TNL::Containers::Vector< double, Device > eigenProduct( size );
      eigenProduct.forAllElements(
         [ eigen ] __cuda_callable__( int i, double& value )
         {
            value = eigen.second[ i ] * eigen.first;
         } );
      double error = TNL::maxNorm( eigenProduct - matrixEigenvector );
      std::cout << "Max error = " << error << std::endl;
      end = std::chrono::high_resolution_clock::now();
      elapsed = end - start;
      std::cout << "Elapsed time rest: " << elapsed.count() << " seconds" << std::endl << std::endl;
   }
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of eigenvalue computation by power iteration method on CPU:" << std::endl;
   bigEigenExample< TNL::Devices::Host >();

   std::cout << "Example of eigenvalue computation by power iteration method on sequential device:" << std::endl;
   bigEigenExample< TNL::Devices::Sequential >();

#ifdef __CUDACC__
   std::cout << "Example of eigenvalue computation by power iteration method on CUDA GPU:" << std::endl;
   bigEigenExample< TNL::Devices::Cuda >();
#endif
}
