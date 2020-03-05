
#include "Matrix/Matrix.h"
#include "gem/GEM.h"
#include "TNL/tnl-dev/src/TNL/Math.h"
#include <typeinfo> // type printf
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
        const String& matrixName,  const String& vectorName, const int verbose );


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
  
  readMatrixVector( matrix, vector, matrixName, vectorName, verbose );
  VectorType vectorResult( matrix.getNumRows() );
  
  // Computation
  double* time;
  time = new double[ loops ];
  double error = -1;
  
  if( verbose > 1 )
    cout << "Starting computation on " << device << endl;
  for( int i = 0; i < loops; i++ )
  {
    MatrixType matrixComp = matrix;
    VectorType vectorComp( vector );
    vectorResult.setValue( 0 );
    
    GEM< Real, Device, Index > gem( matrixComp, vectorComp );

    std::clock_t start;
    double duration;

    start = std::clock();
    
    if( verbose > 1 )
      cout << "starting computation number " << i+1 << endl;
    gem.solve( vectorResult, pivoting, verbose );

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    time[i] = duration;
      
    if( vectorName == "none" )
    {
      double l2norm = 0;
      for( int j = 0; j < vectorResult.getSize(); j++ )
        l2norm += (vectorResult.getElement( j ) - 1)*(vectorResult.getElement( j ) - 1);
      l2norm = std::sqrt(l2norm);
      error = l2norm;

      if( verbose > 1 )
        printf( "Norm in %d calculation is %.4f \n", i+1, l2norm);
    }
  }
  if( verbose > 1 )
    printf("\n ... done!\n");
  
  double timeMean = 0;
  for( int i = 0; i < loops; i++ )
    timeMean += time[i];
  timeMean /= loops;
    
   printf("%20s %15d %15d %20s %15s %15s %10d %15.5f %15.5f\n", matrixName.c_str(),
          matrix.getNumRows(), matrix.getNumNonzeros(), vectorName == "none" ? "-":vectorName.c_str(),
          device.c_str(), typeid(Real).name() == (string)"f" ? "float":"double", loops, timeMean, error);
  
  delete []time;
  
  return vectorResult;
}

template < typename Real, typename Index, typename Device >
void readMatrixVector( Matrix< Real, Device, Index>& matrix, 
        Vector< Real, Device, Index >& vector,
        const String& matrixName,  const String& vectorName, const int verbose )
{
  typedef Matrix< Real, Devices::Host, Index> MatrixHost;
  MatrixHost matrixHost;
  // Get matrix
  Matrices::MatrixReader< MatrixHost > m;
  m.readMtxFile( "./test-matrices/" + matrixName, matrixHost );
  if( verbose > 1 )
    cout << "reading matrix " << matrixName << endl;
  if( verbose > 2 )
    matrixHost.showMatrix();
  matrix = matrixHost;
  
  // Get vector
  Vector< Real, Devices::Host, Index > vectorHost( matrixHost.getNumRows() );
  
  if( vectorName == "none" )
    calculHostVecOne( matrixHost, vectorHost, vectorName );
  else
  {
    readVector( vectorHost, vectorName );
    if( verbose > 1 )
      cout << "reading vector " << vectorName << endl;
  }
  if( verbose > 2 )
    cout << vectorHost << endl;
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
  
  /*ofstream outdata; // outdata is like cin
  
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
  cout << endl;*/
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