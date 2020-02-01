#include <iostream>
#include <cstdlib>

#include "Matrix/Matrix.h"
#include "gem-CPU/GEM.h"
#include "TNL/tnl-dev/src/TNL/Math.h"
#include "gem-GPU/GEMdevice.h"
#include <TNL/Matrices/MatrixReader.h>

#include "TNL/tnl-dev/src/TNL/Cuda/CheckDevice.h"
#include "TNL/tnl-dev/src/TNL/Devices/Cuda.h"
#include "TNL/tnl-dev/src/TNL/Containers/Vector.h"
#include "TNL/tnl-dev/src/TNL/Cuda/MemoryHelpers.h"


#ifdef HAVE_CUDA
__global__ 
void tryKernel( TNL::Containers::Vector< double, TNL::Devices::Cuda, int >* vector, 
        Matrix< double, TNL::Devices::Cuda, int >* matrix )
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; 
  if( i == 0 )
  {
    printf("First display:\n");
    //printf( "%.4f ", vector[0]);
    matrix->showMatrix();
    matrix->setElement( 0,0, 0.);
    printf("Second display:\n");
    matrix->showMatrix();
  }
}
#endif

using namespace TNL;
using namespace TNL::Containers;
using namespace std;

 
int main( int argc, char* argv[] )
{ 
  typedef Matrix< double, Devices::Host, int> MatrixHost;
  typedef Vector< double, Devices::Host, int > VectorHost;
  
  const int size = 9;
  VectorHost dataVec(size*size),host_vector(size), result_vector(size);
  
  MatrixHost matrix( size,size);
  // set matrix for tryal run
  Matrices::MatrixReader<MatrixHost> m;
  m.readMtxFile( "./test-matrices/cage4.mtx", matrix );
  //matrix.showMatrix();
  
  /*matrix.setElement( 0, 0, 8.0 );
  matrix.setElement( 0, 1, -1.0 );
  matrix.setElement( 0, 2, -2.0 );
  
  matrix.setElement( 1, 0, -1.0 );
  matrix.setElement( 1, 1, 7.0 );
  matrix.setElement( 1, 2, -1.0 );
  
  matrix.setElement( 2, 0, -2.0 );
  matrix.setElement( 2, 1, -1.0 );
  matrix.setElement( 2, 2, 9.0 );*/
  
  
  // set vector and output vector
  result_vector.setValue( 0.0 );
  host_vector.setElement(0, 0.0); 
  host_vector.setElement(1, 10.0);
  host_vector.setElement(2, 23.0);
    
  GEM< double, Devices::Host, int > gem( matrix, host_vector );
  gem.solve( result_vector, 0 );
  printf("Host result:\n");
  std::cout << result_vector << std::endl;
  
#ifdef HAVE_CUDA
  typedef Matrix< double, Devices::Cuda, int> MatrixDevice;
  typedef Vector< double, Devices::Cuda, int > VectorDevice;
  VectorDevice device_vector(size), result_vector_dev(size);
  MatrixDevice matrixDev(size,size);
  
  // copy matrix to device
  matrixDev = matrix;
  MatrixDevice* devMat = Cuda::passToDevice( matrixDev);
  
  // copy vectors to device
  result_vector_dev = result_vector;
  device_vector = host_vector;
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
  
  GEMdevice(matrixDev, device_vector, result_vector_dev );
  
  // show results
  matrix = matrixDev;
  printf("Device result:\n");
  matrix.showMatrix();
  cout << device_vector << endl;
  
#endif
  
  
  return EXIT_SUCCESS;
}