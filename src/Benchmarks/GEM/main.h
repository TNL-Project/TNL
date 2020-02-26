#include <iostream>
#include <cstdlib>
#include <string> // input from cmd
#include <ctime> // time of computation mesurement

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

void printHelp(); 

void setInput( int argc, char* argv[], string& matrixName, string& vectorName, int& loops  );

int main( int argc, char* argv[] )
{ 
  string matrixName;
  string vectorName;
  int loops;
  setInput( argc, argv, matrixName, vectorName, loops );
  
  
  typedef Matrix< double, Devices::Host, int> MatrixHost;
  typedef Vector< double, Devices::Host, int > VectorHost;
  
  
  MatrixHost matrix;
  // set matrix for tryal run
  Matrices::MatrixReader<MatrixHost> m;
  m.readMtxFile( "./test-matrices/" + matrixName, matrix );
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
  
  double* timeCPU;
  timeCPU = new double[loops];
  
  printf("Starting computation CPU");
  for( int i = 0; i < loops; i++ )
  {
    MatrixHost matrixComp = matrix;
    VectorHost host_vecComp( host_vector );
    GEM< double, Devices::Host, int > gem( matrixComp, host_vecComp );
    
    std::clock_t start;
    double duration;

    start = std::clock();
    
    gem.solveWithPivoting( result_vector, 0 );

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    timeCPU[i] = duration;
  }
  printf(" ... done!\n");
  /*printf("Host result:\n");
  std::cout << result_vector << std::endl;*/
  
#ifdef HAVE_CUDA
  typedef Matrix< double, Devices::Cuda, int> MatrixDevice;
  typedef Vector< double, Devices::Cuda, int > VectorDevice;
  VectorDevice result_vector_dev(matrix.getNumRows());
  
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
  
  double* timeGPU;
  timeGPU = new double[loops];
  
  printf("Starting computation GPU");
  for( int i = 0; i < loops; i++ )
  {
    MatrixDevice matrixComp; matrixComp = matrix;
    VectorDevice device_vecComp(matrix.getNumRows()); 
    device_vecComp = host_vector;
    
    std::clock_t start;
    double duration;

    start = std::clock();
    GEMdevice(matrixComp, device_vecComp, result_vector_dev );

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    timeGPU[i] = duration;
  }
  printf(" ... done!\n\n");
  
  
  // show results
  /*matrix = matrixDev;
  //printf("Device result:\n");
  //matrix.showMatrix();
  //cout << result_vector_dev << endl;*/
  
#ifdef COMPARE_RESULTS
  double error = 0.0;
  std::cout << "Results:\nrow:"<< setw(20) <<  "Host" << setw(20) <<  "Device" << setw(20) << "Error" << std::endl;
  for( int i = 0; i < matrix.getNumRows(); i++ )
  {
    double errorPom = ( result_vector.getElement(i) - result_vector_dev.getElement(i) ) *
            ( result_vector.getElement(i) - result_vector_dev.getElement(i) );
    std::cout << i << ": " << setw(20) << result_vector.getElement(i) << setw(20) << result_vector_dev.getElement(i) << setw(20) << errorPom << std::endl;
    error += errorPom;
  }
  
  error = std::sqrt(error);
  printf("Difference in L2 norm from Device and Host is %.8f\n", error );
  
  double CPUmean(0), GPUmean(0);
  cout << "Timers: " << endl << "CPU: [ ";
  for( int i = 0; i < loops; i++ )
  {
    CPUmean += timeCPU[i];
    GPUmean += timeGPU[i];
    cout << timeCPU[i] << " ";
  }
  CPUmean = CPUmean/loops;
  GPUmean = GPUmean/loops;
  
  cout << "]" << endl;
  cout << "GPU: [ ";
  for( int i = 0; i < loops; i++ )
    cout << timeGPU[i] << " ";
  cout << "]" << endl;
  cout << "CPU mean time: " <<  CPUmean << endl;
  cout << "GPU mean time: " <<  GPUmean << endl;
#endif
  
#endif
  
  delete []timeCPU;
#ifdef HAVE_CUDA
  delete []timeGPU;
#endif
  return EXIT_SUCCESS;
}


void setInput( int argc, char* argv[], string& matrixName, string& vectorName, int& loops )
{  
  if( argc == 1 )
  {
    string pom("matice6.mtx");
    matrixName = pom; 
    string pom1("vec1.txt");
    vectorName = pom1;
    loops = 10;
  }
  
  for(int i = 1; i < argc; i = i+2)
  {
    cout<< argc << " " << argv[i] <<endl;
    
    if( argc != 7 )
    {
      printf( "You need to put all arguments in function call\n");
      printHelp();
      return;
    }
    
    if( argv[1] != (char*)"--input-matrix" || argv[3] != (char*)"--input-vector" || argv[5] != (char*)"--loops" )
    {
      cout << "You need to set all parameters in the same order like help." << endl;
      printHelp();
    }
    string pom(argv[2]);
    matrixName = pom;
    string pom1(argv[4]);
    vectorName = pom1;
    loops = stoi(argv[6]);
    
  }
  
  cout << "Setting values: \nMatrix ... " << matrixName << endl
          << "Vector ... " << vectorName << endl
          << "Loops ... " << loops << endl << endl;
}

void printHelp()
{
  cout << "Parameter:" << setw(40) << "descriptioun" << endl;
  cout << "--input-matrix" << setw(60) << ".mtx file placed in test-matrices foulder." << endl;
  cout << "--input-vector" << setw(60) << ".txt file placed in test-matrices foulder." << endl;
  cout << "--loops" << setw(92) << "int number of loops of calculation for computation time mesurement." << endl;
}