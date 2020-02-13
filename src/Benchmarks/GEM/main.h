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

#define COMPARE_RESULTS true;

using namespace TNL;
using namespace TNL::Containers;
using namespace std;

 
int main( int argc, char* argv[] )
{ 
  typedef Matrix< double, Devices::Host, int> MatrixHost;
  typedef Vector< double, Devices::Host, int > VectorHost;
  
  
  MatrixHost matrix(236,236);
  // set matrix for tryal run
  Matrices::MatrixReader<MatrixHost> m;
  m.readMtxFile( "./test-matrices/matice2.mtx", matrix );
  VectorHost host_vector(matrix.getNumRows()), result_vector(matrix.getNumRows());
  //matrix.showMatrix();
  
  /*matrix.setElement( 0, 0, 8.0 );
  matrix.setElement( 0, 1, -1.0 );
  matrix.setElement( 0, 2, -2.0 );
  
  matrix.setElement( 1, 0, -1.0 );
  matrix.setElement( 1, 1, 7.0 );
  matrix.setElement( 1, 2, -1.0 );
  
  matrix.setElement( 2, 0, -2.0 );
  matrix.setElement( 2, 1, -1.0 );
  matrix.setElement( 2, 2, 9.0 );
  // set vector and output vector
  host_vector.setElement(0, 0.0);
  host_vector.setElement(1, 10.0);
  host_vector.setElement(2, 23.0);*/
  
  
  // set vector and output vector MATICE6
  host_vector.setValue(0);
  host_vector.setElement(0, 49.0);
  host_vector.setElement(1, 45.0);
  host_vector.setElement(2, 34.0);
  host_vector.setElement(3, 45.0);
  host_vector.setElement(4, 46.0);
  host_vector.setElement(5, 51.0);
  
  
  // set vector and output vector MATICE9
  /*
  result_vector.setValue( 0.0 );
  host_vector.setElement(0, 105.0);
  host_vector.setElement(1, 105.0);
  host_vector.setElement(2, 93.0);
  host_vector.setElement(3, 98.0);
  host_vector.setElement(4, 96.0);
  host_vector.setElement(5, 121.0);
  host_vector.setElement(6, 80.0);
  host_vector.setElement(7, 107.0);
  host_vector.setElement(8, 113.0);*/
    
  GEM< double, Devices::Host, int > gem( matrix, host_vector );
  gem.solveWithPivoting( result_vector, 0 );
  /*printf("Host result:\n");
  std::cout << result_vector << std::endl;*/
  
#ifdef HAVE_CUDA
  typedef Matrix< double, Devices::Cuda, int> MatrixDevice;
  typedef Vector< double, Devices::Cuda, int > VectorDevice;
  VectorDevice device_vector(matrix.getNumRows()), result_vector_dev(matrix.getNumRows());
  MatrixDevice matrixDev;
  
  // copy matrix to device
  matrixDev = matrix;
  
  // copy vectors to device
  device_vector = host_vector;
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
  
  GEMdevice(matrixDev, device_vector, result_vector_dev );
  
  // show results
  /*matrix = matrixDev;
  //printf("Device result:\n");
  //matrix.showMatrix();
  //cout << result_vector_dev << endl;*/
  
#ifdef COMPARE_RESULTS
  double error = 0.0;
  std::cout << "Results:\nrow: Host Device Error" << std::endl;
  for( int i = 0; i < matrix.getNumRows(); i++ )
  {
    double errorPom = ( result_vector.getElement(i) - result_vector_dev.getElement(i) ) *
            ( result_vector.getElement(i) - result_vector_dev.getElement(i) );
    std::cout << i << ": " << result_vector.getElement(i) << " " << result_vector_dev.getElement(i) << " " << errorPom << std::endl;
    error += errorPom;
  }
  
  error = std::sqrt(error);
  printf("Difference in L2 norm from Device and Host is %.8f\n", error );
#endif
  
#endif
  
  
  return EXIT_SUCCESS;
}