#include <chrono>
#include <thread>

//#include "Matrix/Matrix.h"
#include <TNL/Solvers/Linear/GEM.h>
#include <TNL/Math.h>
#include <typeinfo>  // type printf
#include <TNL/Matrices/MatrixReader.h>

#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>

#ifdef HAVE_MPI
   #include <TNL/Communicators/MpiCommunicator.h>
   #include <mpi.h>
#endif

#define COMPARE_RESULTS true;

using namespace TNL;
using namespace TNL::Containers;
using namespace std;

template< typename Real, typename Index >
void
calculHostVecOne( Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix,
                  Vector< Real, Devices::Host, Index >& vector,
                  const String& vectorName );

#ifdef HAVE_MPI
template< typename Real, typename Index >
void
cutMatrixVectorMPI( Matrix< Real, Devices::Host, Index >& matrix, Vector< Real, Devices::Host, Index >& vector );
#endif

template< typename Real, typename Index >
void
readVector( Vector< Real, Devices::Host, Index >& vector_host, const String& vectorName );

template< typename Real, typename Index >
void
readMatrixVector( TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix,
                  Vector< Real, Devices::Host, Index >& vector,
                  const String& matrixName,
                  const String& vectorName,
                  Index& rows,
                  Index& nonzeros,
                  const int verbose );

template< typename Real, typename Index, typename Device >
Vector< Real, Device, Index >
runGEM( const String& matrixName,
        const String& vectorName,
        const int loops,
        const int verbose,
        const String& device,
        const String& pivoting )
{
   typedef TNL::Matrices::DenseMatrix< Real, Device, Index > MatrixType;
   typedef Vector< Real, Device, Index > VectorType;
   typedef TNL::Matrices::DenseMatrix< Real, Devices::Host, Index > MatrixTypeHost;
   typedef Vector< Real, Devices::Host, Index > VectorTypeHost;

   int processID = 0;  // MPI processID, without mpi == 0
#ifdef HAVE_MPI
   MPI_Comm_rank( MPI_COMM_WORLD, &processID );
   Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
   MatrixTypeHost matrixHost;
   VectorTypeHost vectorHost;
   MatrixType matrix;
   VectorType vector;
   VectorType vectorResult;

   Index rows, nonzeros;
   readMatrixVector( matrixHost, vectorHost, matrixName, vectorName, rows, nonzeros, verbose );

   // Computation
   double* time;
   time = new double[ loops ];
   double error = -1;

   if( verbose > 1 )
      cout << "Starting computation on " << device << endl;

   for( int i = 0; i < loops; i++ ) {
#ifdef HAVE_MPI
      Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
      //readMatrixVector( matrixHost, vectorHost, matrixName, vectorName, rows, nonzeros, verbose );
      matrix.reset();
      vector.reset();
      matrix = matrixHost;
      vector = vectorHost;
      vectorResult.setSize( rows );
      vectorResult.setValue( 0 );
      TNL::Solvers::Linear::GEM< Real, Device, Index > gem( matrix, vector );

#ifdef HAVE_MPI
      Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
      double duration;
      std::clock_t start;
      start = std::clock();
      if( verbose > 1 && processID == 0 )
         cout << "starting computation number " << i + 1 << endl;

      gem.solve( vectorResult, pivoting, verbose );

      duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

      if( processID == 0 ) {
         time[ i ] = duration;

         if( vectorName == "none" ) {
            double l2norm = 0;
            for( int j = 0; j < vectorResult.getSize(); j++ )
               l2norm += ( vectorResult.getElement( j ) - 1 ) * ( vectorResult.getElement( j ) - 1 );
            l2norm = std::sqrt( l2norm );
            error = l2norm;

            if( verbose > 1 )
               printf( "Norm in %d calculation is %.4f\n", i + 1, l2norm );
         }
      }
   }
   if( verbose > 1 && processID == 0 )
      printf( "\n ... done!\n" );

   double timeMean = 0;
   for( int i = 0; i < loops; i++ )
      timeMean += time[ i ];
   timeMean /= loops;

   if( processID == 0 )
      printf( "%20s %15s %15s %10d %20s & %15d & %15.3f & %15.3f\n",
              vectorName == "none" ? "-" : vectorName.c_str(),
              device.c_str(),
              typeid( Real ).name() == ( string ) "f" ? "float" : "double",
              loops,
              matrixName.c_str(),
              rows,
              timeMean,
              error );

   delete[] time;

#ifdef HAVE_MPI
   Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
   matrix.reset();
   if( processID == 0 ) {
      //printf("%d: returning\n", processID );
      return vectorResult;
   }
   else {
      //("%d: returning\n", processID );
      //vectorResult.setValue( 0 );
      return vectorResult;
   }
}

template< typename Real, typename Index >
void
readMatrixVector( TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix,
                  Vector< Real, Devices::Host, Index >& vector,
                  const String& matrixName,
                  const String& vectorName,
                  Index& rows,
                  Index& nonzeros,
                  const int verbose )
{
   if( verbose > 1 )
      cout << "reading matrix " << matrixName << endl;
   // Get matrix
   Matrices::MatrixReader< TNL::Matrices::DenseMatrix< Real, Devices::Host, Index > > m;
   m.readMtx( "./test-matrices/" + matrixName, matrix, verbose );
   rows = matrix.getRows();
   //nonzeros = matrix.getNonzeros();
   vector.setSize( rows );

   // Get vector
   if( vectorName == "none" ) {
      calculHostVecOne( matrix, vector, vectorName );
   }
   else {
      readVector( vector, vectorName );
      if( verbose > 1 )
         cout << "reading vector " << vectorName << endl;
   }

#ifdef HAVE_MPI
   cutMatrixVectorMPI( matrix, vector );
#endif
   if( verbose > 2 )
      std::cout << matrix << std::endl; //matrix.showMatrix();

   if( verbose > 2 )
      cout << vector << endl;
}

template< typename Real, typename Index >
void
calculHostVecOne( TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix,
                  Vector< Real, Devices::Host, Index >& vector,
                  const String& vectorName )
{
   for( int i = 0; i < matrix.getRows(); i++ ) {
      Real pom = 0;
      for( int j = 0; j < matrix.getColumns(); j++ ) {
         pom += matrix.getElement( i, j );
      }
      vector.setElement( i, pom );
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

template< typename Real, typename Index >
void
readVector( Vector< Real, Devices::Host, Index >& vector_host, const String& vectorName )
{
   ifstream inFile;
   Real x;
   inFile.open( "./test-matrices/" + vectorName );
   if( ! inFile ) {
      cout << "Unable to open file" << endl;
      return;
   }

   int i = 0;
   while( inFile >> x ) {
      vector_host[ i ] = x;
      i++;
   }
   inFile.close();
}

#ifdef HAVE_MPI
template< typename Real, typename Index >
void
cutMatrixVectorMPI( TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix, Vector< Real, Devices::Host, Index >& vector )
{
   TNL::Matrices::DenseMatrix< Real, Devices::Host, Index > matrixTemp;
   Vector< Real, Devices::Host, Index > vectorTemp;

   int processID;
   int numOfProcesses;

   MPI_Comm_rank( MPI_COMM_WORLD, &processID );
   MPI_Comm_size( MPI_COMM_WORLD, &numOfProcesses );

   /*if( processID == 0 )
   {
     printf( "%d: %d\n", numOfProcesses, processID );
     matrix.showMatrix();
     cout << vector << endl;
   }
   */

   Index numRowsCUT = TNL::roundUpDivision( matrix.getNumRows(), numOfProcesses );
   matrixTemp.setDimensions( numRowsCUT, matrix.getNumRows() );
   vectorTemp.setSize( numRowsCUT );

   //printf( "%d: %d num of rows = %d\n", numOfProcesses, processID, numRowsCUT );

   for( int j = 0; j < matrix.getNumColumns(); j++ ) {
      for( int i = 0; i < numRowsCUT; i++ ) {
         if( i + numRowsCUT * processID < matrix.getNumRows() ) {
            matrixTemp.setElement( i, j, matrix.getElement( i + numRowsCUT * processID, j ) );
         }
         else {
            matrixTemp.setElement( i, j, 0 );
         }
      }
      //if( j % 1000 == 0 )
      //  printf("%d: cutting col %d\n", processID, j );
   }
   for( int i = 0; i < numRowsCUT; i++ ) {
      if( i + numRowsCUT * processID < matrix.getNumRows() ) {
         vectorTemp.setElement( i, vector.getElement( i + numRowsCUT * processID ) );
      }
      else {
         vectorTemp.setElement( i, 0 );
      }
   }
   matrix.reset();
   vector.reset();
   matrix = matrixTemp;
   vector = vectorTemp;
}
#endif
