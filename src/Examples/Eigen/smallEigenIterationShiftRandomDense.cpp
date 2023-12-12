#include "TNL/Cuda/CudaCallable.h"
#include "TNL/Devices/Sequential.h"
#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIterationShift.h>
#include <TNL/Algorithms/fillRandom.h>
#include <ostream>

template< typename Device >
void
smallEigenExample()
{
   TNL::Containers::Vector< double, Device > result( 3 );
   TNL::Matrices::DenseMatrix< double, Device > eigenMatrix{ { 1, 0, 3 }, { 3, -2, -1 }, { 1, -1, 1 } };
   std::cout << "Matrix for eigen computation:" << std::endl << eigenMatrix << std::endl;
   std::pair< double, TNL::Containers::Vector< double, Device > > eigen =
      TNL::Matrices::Eigen::powerIteration< double, Device >( eigenMatrix, 1e-7 );
   std::cout << "Eigenvalue: " << eigen.first << "\nEigenvector:" << eigen.second << std::endl;

   eigenMatrix.vectorProduct( eigen.second, result );
   std::cout << "Test by multiplying matrix with his eigenvector: " << result << std::endl;

   eigen =
      TNL::Matrices::Eigen::powerIterationShift< double, Device >( eigenMatrix, 1e-7, -3 );
   std::cout << "Eigenvalue: " << eigen.first << "\nEigenvector:" << eigen.second << std::endl;

   eigenMatrix.vectorProduct( eigen.second, result );
   std::cout << "Test by multiplying matrix with his eigenvector: " << result << std::endl;
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
