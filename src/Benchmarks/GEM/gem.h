
#include "Matrix/Matrix.h"
#include "gem/GEM.h"
#include "TNL/tnl-dev/src/TNL/Math.h"
//#include "gem/GEMdevice.h"
#include <TNL/Matrices/MatrixReader.h>

#include "TNL/tnl-dev/src/TNL/Cuda/CheckDevice.h"
#include "TNL/tnl-dev/src/TNL/Devices/Cuda.h"
#include "TNL/tnl-dev/src/TNL/Containers/Vector.h"
#include "TNL/tnl-dev/src/TNL/Cuda/MemoryHelpers.h"
#include "TNL/tnl-dev/src/TNL/Devices/Host.h"

#define COMPARE_RESULTS true;

// Prihodit GEMdeice do GEM a dělat dělení na device až tam -> přidat parametr pivoting

using namespace TNL;
using namespace TNL::Containers;
using namespace std;


template < typename Real, typename Index >
void calculHostVecOne( Matrix< Real, Devices::Host, Index >& matrix,
        Vector< Real, Devices::Host, Index >& vector, const String& vectorName );

template < typename Real, typename Index >
void readVector( Vector< Real, Devices::Host, Index >& vector_host, const String& vectorName );


template < typename Real, typename Index, typename Device >
void readMatrixVector( Matrix< Real, Device, Index>& matrix, 
        Vector< Real, Device, Index >& vector,
        const String& matrixName,  const String& vectorName );


template < typename Real,
        typename Index,
        typename Device >
Vector< Real, Device, Index > runGEM( const String& matrixName, const String& vectorName, const int loops,
        const int verbose, const String& device, const String& pivoting )
{  
  typedef Matrix< Real, Device, Index > MatrixType;
  typedef Vector< Real, Device, Index > VectorType;
  
  MatrixType matrix;
  VectorType vector;
  
  readMatrixVector( matrix, vector, matrixName, vectorName );
  VectorType vectorResult( matrix.getNumRows() );
  
  // Computation
  double* time;
  time = new double[ loops ];
  
  std::cout << "Starting computation on " << device << endl;
  for( int i = 0; i < loops; i++ )
  {
    MatrixType matrixComp = matrix;
    VectorType vectorComp( vector );
    vectorComp.setValue( 0 );
    vectorResult.setValue( 0 );
    
    GEM< Real, Device, Index > gem( matrixComp, vectorComp );

    std::clock_t start;
    double duration;

    start = std::clock();
    
    cout << "starting computation number " << i+1 << endl;
    gem.solve( vectorResult, pivoting, 0 );

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    time[i] = duration;
      
    double l2norm = 0;
    for( int j = 0; j < matrix.getNumRows(); j++ )
      l2norm += (vectorResult.getElement( i ) - 1)*(vectorResult.getElement( i ) - 1);
    l2norm = std::sqrt(l2norm);
    printf( " %.4f ", l2norm);
  }
  printf("\n ... done!\n");
  
  delete []time;
  
  return vectorResult;
  
 /*
//#ifdef HAVE_CUDA
  
  
  
  
#ifdef COMPARE_RESULTS
  double error = 0.0;
  //std::cout << "Results:\nrow:"<< setw(20) <<  "Host" << setw(20) <<  "Device" << setw(20) << "Error" << std::endl;
  for( int i = 0; i < matrix.getNumRows(); i++ )
  {
    double errorPom = ( result_vector.getElement(i) - result_vector_dev.getElement(i) ) *
            ( result_vector.getElement(i) - result_vector_dev.getElement(i) );
    //std::cout << i << ": " << setw(20) << result_vector.getElement(i) << setw(20) << result_vector_dev.getElement(i) << setw(20) << errorPom << std::endl;
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
  return;*/
}

template < typename Real, typename Index, typename Device >
void readMatrixVector( Matrix< Real, Device, Index>& matrix, 
        Vector< Real, Device, Index >& vector,
        const String& matrixName,  const String& vectorName )
{
  typedef Matrix< Real, Devices::Host, Index> MatrixHost;
  MatrixHost matrixHost;
  // Get matrix
  Matrices::MatrixReader< MatrixHost > m;
  m.readMtxFile( "./test-matrices/" + matrixName, matrixHost );
  cout << "reading matrix " << matrixName << endl;
  //matrixHost.showMatrix();
  matrix = matrixHost;
  
  // Get vector
  Vector< Real, Devices::Host, Index > vectorHost( matrixHost.getNumRows() );
  
  if( vectorName == "none" )
    calculHostVecOne( matrixHost, vectorHost, vectorName );
  else
    readVector( vectorHost, vectorName );
  cout << "reading vector " << vectorName << endl;
  vector = vectorHost;
  
}

template < typename Real, typename Index >
void calculHostVecOne( Matrix< Real, Devices::Host, Index >& matrix,
        Vector< Real, Devices::Host, Index >& vector, const String& vectorName )
{
  for( int i = 0; i < matrix.getNumRows(); i++ )
  {
    Real pom = 0;
    for( int j = 0; j < matrix.getNumColumns(); j++ )
    {
      pom += matrix.getElement(i,j);
    }
    vector[i] = pom;
  }
  cout << endl;
  
  ofstream outdata; // outdata is like cin
  
  outdata.open("./test-matrices/" + vectorName ); // opens the file
  if( !outdata ) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }
  
  for( int i = 0; i < vector.getSize(); i++ )
  {
    outdata << vector[i] << endl;
  }
  outdata.close();
  cout << endl;
}

template < typename Real, typename Index >
void readVector( Vector< Real, Devices::Host, Index >& vector_host, const String& vectorName )
{
  ifstream inFile;
  Real x;
  inFile.open("./test-matrices/" + vectorName );
  if (!inFile) {
      cout << "Unable to open file" << endl;
      return;
  }

  int i = 0;
  while ( inFile >> x ) {
    vector_host[i] = x;
    i++;
  }
  inFile.close();
}