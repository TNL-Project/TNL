#include <chrono>
#include <thread>

#include <TNL/Benchmarks/Benchmark.h>
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

template< typename Real, typename Index >
void
calculHostVecOne( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                  TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector,
                  const TNL::String& vectorName );

#ifdef HAVE_MPI
template< typename Real, typename Index >
void
cutMatrixVectorMPI( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                    TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector );
#endif

template< typename Real, typename Index >
void
readVector( TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector_host, const TNL::String& vectorName );

template< typename Real, typename Index >
void
readMatrixVector( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                  TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector,
                  const TNL::String& matrixName,
                  const TNL::String& vectorName,
                  Index& rows,
                  Index& nonzeros,
                  const int verbose );

template< typename Real, typename Index, typename Device >
TNL::Containers::Vector< Real, Device, Index >
benchmarkGEM( TNL::Config::ParameterContainer& parameters )
{
   using MatrixType = TNL::Matrices::DenseMatrix< Real, Device, Index >;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >;
   using HostVectorType = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;
   using MatrixPointer = std::shared_ptr< MatrixType >;

   auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
   const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
   const auto outputMode = parameters.getParameter< TNL::String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );
   TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

   //   int processID = 0;  // MPI processID, without mpi == 0
   //#ifdef HAVE_MPI
   //   MPI_Comm_rank( MPI_COMM_WORLD, &processID );
   //   Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
   //#endif

   HostMatrixType matrixHost;
   HostVectorType vectorHost;
   MatrixPointer matrix;
   VectorType b, x;

   TNL::Matrices::MatrixReader< MatrixType > reader;
   reader.readMtx( file_matrix, *matrix );
   x.setSize( matrix->getRows() );
   x = 1;
   b.setSize( matrix->getRows() );
   matrix->vectorProduct( x, b );

   //Index rows, nonzeros;
   //readMatrixVector( matrixHost, vectorHost, matrixName, vectorName, rows, nonzeros, verbose );

   // Computation
   double* time;
   time = new double[ loops ];
   double error = -1;

   if( verbose > 1 )
      std::cout << "Starting computation on " << device << std::endl;

   for( int i = 0; i < loops; i++ ) {
#ifdef HAVE_MPI
      Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
      //readMatrixVector( matrixHost, vectorHost, matrixName, vectorName, rows, nonzeros, verbose );
      matrix.reset();
      *matrix = matrixHost;
      b = vectorHost;
      x.setSize( rows );
      x.setValue( 0 );
      TNL::Solvers::Linear::GEM< MatrixType > gem;
      gem.setMatrix( matrix );
      gem.setPivoting( pivoting == "true" );

#ifdef HAVE_MPI
      Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
      double duration;
      std::clock_t start;
      start = std::clock();
      if( verbose > 1 && processID == 0 )
         std::cout << "starting computation number " << i + 1 << std::endl;

      gem.solve( b, x );

      duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

      if( processID == 0 ) {
         time[ i ] = duration;

         if( vectorName == "none" ) {
            error = l2Norm( x - 1 );
            /*double l2norm = 0;
            for( int j = 0; j < vectorResult.getSize(); j++ )
               l2norm += ( vectorResult.getElement( j ) - 1 ) * ( vectorResult.getElement( j ) - 1 );
            l2norm = std::sqrt( l2norm );
            error = l2norm;*/

            if( verbose > 1 )
               std::cout << "Error in " << i + 1 << " calculation is " << error << std::endl;
         }
      }
   }
   if( verbose > 1 && processID == 0 )
      std::cout << "\n ... done!\n";

   double timeMean = 0;
   for( int i = 0; i < loops; i++ )
      timeMean += time[ i ];
   timeMean /= loops;

   if( processID == 0 )
      //printf( "%20s %15s %15s %10d %20s & %15d & %15.3f & %15.3f\n",
      std::cout << ( vectorName == "none" ? "-" : vectorName.c_str() ) << " " << device.c_str() << " "
                << ( typeid( Real ).name() == ( std::string ) "f" ? "float" : "double" ) << " " << loops << " " << matrixName
                << " " << rows << " " << timeMean << " " << error << std::endl;

   delete[] time;

#ifdef HAVE_MPI
   Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
   matrix.reset();
   if( processID == 0 ) {
      //printf("%d: returning\n", processID );
      return x;
   }
   else {
      //("%d: returning\n", processID );
      //vectorResult.setValue( 0 );
      return x;
   }
}

template< typename Real, typename Index >
void
readMatrixVector( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                  TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector,
                  const TNL::String& matrixName,
                  const TNL::String& vectorName,
                  Index& rows,
                  Index& nonzeros,
                  const int verbose )
{
   if( verbose > 1 )
      std::cout << "reading matrix " << matrixName << std::endl;
   // Get matrix
   TNL::Matrices::MatrixReader< TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index > > m;
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
         std::cout << "reading vector " << vectorName << std::endl;
   }

#ifdef HAVE_MPI
   cutMatrixVectorMPI( matrix, vector );
#endif
   if( verbose > 2 )
      std::cout << matrix << std::endl;  //matrix.showMatrix();

   if( verbose > 2 )
      std::cout << vector << std::endl;
}

template< typename Real, typename Index >
void
calculHostVecOne( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                  TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector,
                  const TNL::String& vectorName )
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
readVector( TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector_host, const TNL::String& vectorName )
{
   std::ifstream inFile;
   Real x;
   inFile.open( "./test-matrices/" + vectorName );
   if( ! inFile ) {
      std::cout << "Unable to open file" << std::endl;
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
cutMatrixVectorMPI( TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
                    TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& vector )
{
   TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index > matrixTemp;
   TNL::Containers::Vector< Real, TNL::Devices::Host, Index > vectorTemp;

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
