#include "TNL/Backend.h"
#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Algorithms/fillRandom.h>
#include <ostream>

template< typename Device >
void
smallEigenExample()
{
   TNL::Containers::Vector< double, Device > result( 3 );
   TNL::Matrices::DenseMatrix< double, Device > eigenMatrix{ { 1, 1, 0 }, { 0, 1, 1 }, { 0, 0, 1 } };
   std::cout << "Matrix for eigen computation:" << std::endl << eigenMatrix << std::endl;
   std::pair< double, TNL::Containers::Vector< double, Device > > eigen =
      TNL::Matrices::Eigen::powerIteration< double, Device >( eigenMatrix, 1e-7 );
   std::cout << "Eigenvalue: " << eigen.first << "\nEigenvector:" << eigen.second << std::endl;

   eigenMatrix.vectorProduct( eigen.second, result );
   std::cout << "Test by multiplying matrix with his eigenvector: " << result << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > randomMatrix( 10, 10 );
   TNL::Algorithms::fillRandom< Device >( randomMatrix.getValues().getData(), 10 * 10, (double)0, (double)1 );
   std::cout << "Random matrix for eigen computation:" << std::endl << randomMatrix << std::endl;
   eigen = TNL::Matrices::Eigen::powerIteration< double, Device >( randomMatrix, 1e-7 );
   std::cout << "Eigenvalue: " << eigen.first << "\nEigenvector:" << eigen.second << std::endl;
   TNL::Containers::Vector< double, Device > resultRandom( 10 );
   randomMatrix.vectorProduct( eigen.second, resultRandom );
   std::cout << "Test by multiplying random matrix with his eigenvector: " << resultRandom << std::endl;
   TNL::Containers::Vector< double, Device > resultRandom2( 10 );
   resultRandom2.forAllElements(
      [ eigen ] __cuda_callable__( int i, double& value )
      {
         value = eigen.second[i]*eigen.first;
      } );
   std::cout << "Test by multiplying eigenvector with eigenvalue:" << resultRandom2 << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of eigenvalue computation by power iteration method on CPU:" << std::endl;
   smallEigenExample< TNL::Devices::Host >();

   std::cout << "Example of eigenvalue computation by power iteration method on sequential device:" << std::endl;
   smallEigenExample< TNL::Devices::Sequential >();

#ifdef __CUDACC__
   std::cout << "Example of eigenvalue computation by power iteration method on CUDA GPU:" << std::endl;
   smallEigenExample< TNL::Devices::Cuda >();
#endif
}
