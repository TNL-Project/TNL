#include "TNL/Algorithms/Segments/ElementsOrganization.h"
#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Cuda/CudaCallable.h"
#include "TNL/Devices/Sequential.h"
#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Algorithms/fillRandom.h>
#include <ostream>

/*
*/

template< typename Device >
void
bigEigenExample()
{
   int size = 2000;
   std::cout << "Example with RowMajorOrder:" << std::endl;
   for( int i = 0; i < 1; i++ ) {
      TNL::Matrices::DenseMatrix< double, Device, int, TNL::Algorithms::Segments::RowMajorOrder > randomMatrix( size, size );
      auto start = std::chrono::high_resolution_clock::now();
      TNL::Algorithms::fillRandom< Device >( randomMatrix.getValues().getData(), size * size, (double) 0, (double) 1 );
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

   std::cout << "Example with ColumnMajorOrder:" << std::endl;
   for( int i = 0; i < 1; i++ ) {
      TNL::Matrices::DenseMatrix< double, Device, int, TNL::Algorithms::Segments::ColumnMajorOrder > randomMatrix( size, size );
      auto start = std::chrono::high_resolution_clock::now();
      TNL::Algorithms::fillRandom< Device >( randomMatrix.getValues().getData(), size * size, (double) 0, (double) 1 );
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > elapsed = end - start;
      std::cout << "Elapsed time fillRandom: " << elapsed.count() << " seconds" << std::endl;
      std::pair< double, TNL::Containers::Vector< double, Device > > eigen =
         TNL::Matrices::Eigen::powerIteration< double, Device >( randomMatrix, 1e-7 );
      start = std::chrono::high_resolution_clock::now();
      elapsed = start - end;
      std::cout << "Elapsed time powerIteration: " << elapsed.count() << " seconds" << std::endl;
      //std::cout << "Eigenvalue: " << eigen.first << "\nEigenvector:" << eigen.second << std::endl;
      TNL::Containers::Vector< double, Device > matrixEigenvector( size );
      randomMatrix.vectorProduct( eigen.second, matrixEigenvector );
      //std::cout << "Test by multiplying random matrix with his eigenvector: " << resultRandom << std::endl;
      TNL::Containers::Vector< double, Device > eigenProduct( size );
      eigenProduct.forAllElements(
         [ eigen ] __cuda_callable__( int i, double& value )
         {
            value = eigen.second[ i ] * eigen.first;
         } );
      //std::cout << "Test by multiplying eigenvector with eigenvalue:" << resultRandom2 << std::endl;
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
