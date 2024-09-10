// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/TypeInfo.h>
#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Matrices/Eigen/ShiftedPowerIteration.h>
#include <TNL/Matrices/Eigen/QRalgorithm.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Algorithms/fillRandom.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include <cstring>
#include <iostream>
#include <string>

#include "EigenBenchmark.h"

using namespace TNL;
using namespace Benchmarks;
using namespace Containers;

template< typename VectorType >
VectorType
generateVector( const int& size )
{
   using PrecisionType = typename VectorType::RealType;
   using Device = typename VectorType::DeviceType;
   VectorType vector( size );
   TNL::Algorithms::fillRandom< Device >( vector.getData(), vector.getSize(), (PrecisionType) -1, (PrecisionType) 1 );
   return vector;
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
void
benchmark_pi( Benchmark<>& benchmark, MatrixType& matrix, VectorType& initialVecOrig )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   Vector< double, Device > doubleEigenvector( matrix.getColumns() );
   doubleMatrix = matrix;
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      double error = 0;
      int iterations = 0;
      PrecisionType eigenvalue = 0;
      VectorType eigenvector( matrix.getColumns() );
      VectorType initialVec( matrix.getColumns() );
      eigenvector.setValue( 0 );
      initialVec.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         Vector< double, Device > matrixEigenvector( matrix.getColumns() );
         doubleEigenvector = eigenvector;
         doubleMatrix.vectorProduct( doubleEigenvector, matrixEigenvector );
         error += TNL::maxNorm( ( eigenvalue * doubleEigenvector ) - matrixEigenvector );
         iterations += iter;
         initialVec = initialVecOrig;
      };
      auto testFunction = [ & ]()
      {
         std::tie( eigenvalue, eigenvector, iter ) =
            Matrices::Eigen::powerIteration< MatrixType >( matrix, epsilon, initialVec, 100000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
void
benchmark_spi( Benchmark<>& benchmark, MatrixType& matrix, VectorType& initialVecOrig, const PrecisionType& shiftValue )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   PrecisionType eigenvalue = 0;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   doubleMatrix = matrix;
   for( int i = 1; i < 14; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      double error = 0;
      int iterations = 0;
      VectorType eigenvector( matrix.getColumns() );
      VectorType initialVec( matrix.getColumns() );
      eigenvector.setValue( 0 );
      initialVec.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         Vector< double, Device > matrixEigenvector( matrix.getColumns() );
         Vector< double, Device > doubleEigenvector( matrix.getColumns() );
         doubleEigenvector = eigenvector;
         double doubleEigenvalue = eigenvalue;
         doubleMatrix.vectorProduct( doubleEigenvector, matrixEigenvector );
         error += TNL::maxNorm( ( doubleEigenvalue * doubleEigenvector ) - matrixEigenvector );
         iterations += iter;
         initialVec = initialVecOrig;
      };
      auto testFunction = [ & ]()
      {
         std::tie( eigenvalue, eigenvector, iter ) =
            Matrices::Eigen::shiftedPowerIteration< MatrixType >( matrix, epsilon, shiftValue, initialVec, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO, typename VectorType >
void
run_benchmarks_DM( Benchmark<>& benchmark,
                   const std::string& matrixName,
                   int& size,
                   MatrixTypeCMO& matrixCMO,
                   VectorType& initialVecOrig,
                   const PrecisionType& shiftValue )
{
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixName", matrixName },
      { "matrixType", "DM_CMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI0" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig, 0 );
   }

   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixName", matrixName },
      { "matrixType", "DM_RMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_RMO" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI0" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_RMO" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig, 0 );
   }
}

template< typename Device, typename PrecisionType, typename MatrixType, typename VectorType >
void
run_benchmarks_SM( Benchmark<>& benchmark,
                   const std::string& matrixName,
                   const int& size,
                   MatrixType& matrixSM,
                   VectorType& initialVecOrig,
                   const PrecisionType& shiftValue )
{
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixName", matrixName },
      { "matrixType", "SM" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "SM" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "SPI0" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "SM" },
         { "size", std::to_string( size ) },
      } ) );
      benchmark_spi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig, 0 );
   }
}
template< typename PrecisionType >
void
run_benchmarks_file( Benchmark<>& benchmark, const std::string& fileName, PrecisionType shiftValue = 0 )
{
   std::string matrixName = fileName;
   matrixName.erase( matrixName.length() - 4, 4 );
   using MatrixTypeHostSM = Matrices::SparseMatrix< PrecisionType, Devices::Host, int >;
   MatrixTypeHostSM matrixSM;
   TNL::Matrices::MatrixReader< MatrixTypeHostSM >::readMtx( fileName, matrixSM );
   auto size = matrixSM.getColumns();
   using VectorTypeHost = Vector< PrecisionType, Devices::Host >;
   auto initialVecOrig = generateVector< VectorTypeHost >( size );
   run_benchmarks_SM< Devices::Host, PrecisionType >( benchmark, matrixName, size, matrixSM, initialVecOrig, shiftValue );
#ifdef __CUDACC__
   using VectorTypeCuda = Vector< PrecisionType, Devices::Cuda >;
   VectorTypeCuda initialVecOrigCuda( size );
   initialVecOrigCuda = initialVecOrig;
   Matrices::SparseMatrix< double, Devices::Cuda, int > matrixCUDACMO( size, size );
   matrixCUDACMO = matrixSM;
   run_benchmarks_SM< Devices::Cuda, PrecisionType >(
      benchmark, matrixName, size, matrixCUDACMO, initialVecOrigCuda, shiftValue );
#endif
   if( size <= 1600 ) {
      using MatrixTypeHostCMO =
         Matrices::DenseMatrix< PrecisionType, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
      MatrixTypeHostCMO matrixDoubleDM;
      TNL::Matrices::MatrixReader< MatrixTypeHostCMO >::readMtx( fileName, matrixDoubleDM );
      run_benchmarks_DM< Devices::Host, PrecisionType >(
         benchmark, matrixName, size, matrixDoubleDM, initialVecOrig, shiftValue );
#ifdef __CUDACC__
      Matrices::DenseMatrix< PrecisionType, Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >
         matrixCUDADoubleCMO( size, size );
      matrixCUDADoubleCMO = matrixDoubleDM;
      run_benchmarks_DM< Devices::Cuda, PrecisionType >(
         benchmark, matrixName, size, matrixCUDADoubleCMO, initialVecOrigCuda, shiftValue );
#endif
   }
}

void
run_benchmarks( Benchmark<>& benchmark )
{
   //https://sparse.tamu.edu/HB/bcspwr01
   run_benchmarks_file< float >( benchmark, "bcspwr01.mtx", -0.8 );
   run_benchmarks_file< double >( benchmark, "bcspwr01.mtx", -0.8 );
   //https://sparse.tamu.edu/HB/bcsstk01
   run_benchmarks_file< float >( benchmark, "bcsstk01.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstk01.mtx" );
   //http://sparse.tamu.edu/Bai/bfwb62
   run_benchmarks_file< float >( benchmark, "bfwb62.mtx" );
   run_benchmarks_file< double >( benchmark, "bfwb62.mtx" );
   //https://sparse.tamu.edu/Newman/polbooks
   run_benchmarks_file< float >( benchmark, "polbooks.mtx" );
   run_benchmarks_file< double >( benchmark, "polbooks.mtx" );
   //https://sparse.tamu.edu/HB/bcsstm03
   run_benchmarks_file< float >( benchmark, "bcsstm03.mtx", -1000 );
   run_benchmarks_file< double >( benchmark, "bcsstm03.mtx", -1000 );
   //https://sparse.tamu.edu/HB/bcsstm05
   run_benchmarks_file< float >( benchmark, "bcsstm05.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstm05.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk05
   run_benchmarks_file< float >( benchmark, "bcsstk05.mtx", -2603912.5565242684 );
   run_benchmarks_file< double >( benchmark, "bcsstk05.mtx", -2603912.5565242684 );
   //https://sparse.tamu.edu/HB/plat362
   run_benchmarks_file< float >( benchmark, "plat362.mtx" );
   run_benchmarks_file< double >( benchmark, "plat362.mtx" );
   //http://sparse.tamu.edu/Bai/bfwa782
   run_benchmarks_file< float >( benchmark, "bfwa782.mtx" );
   run_benchmarks_file< double >( benchmark, "bfwa782.mtx" );
   //https://sparse.tamu.edu/HB/lshp1561
   run_benchmarks_file< float >( benchmark, "lshp1561.mtx", -2.46630660850344 );
   run_benchmarks_file< double >( benchmark, "lshp1561.mtx", -2.46630660850344 );
   //https://sparse.tamu.edu/HB/bcsstk13 - python no convergence
   run_benchmarks_file< float >( benchmark, "bcsstk13.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstk13.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk21 - python no convergence
   run_benchmarks_file< float >( benchmark, "bcsstk21.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstk21.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk29
   run_benchmarks_file< float >( benchmark, "bcsstk29.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstk29.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk30
   run_benchmarks_file< float >( benchmark, "bcsstk30.mtx", -30.735151383497943 );
   run_benchmarks_file< double >( benchmark, "bcsstk30.mtx", -30.735151383497943 );
   //https://sparse.tamu.edu/HB/bcsstk32
   run_benchmarks_file< float >( benchmark, "bcsstk32.mtx" );
   run_benchmarks_file< double >( benchmark, "bcsstk32.mtx" );
   //https://sparse.tamu.edu/GHS_psdef/s3dkt3m2
   run_benchmarks_file< float >( benchmark, "s3dkt3m2.mtx" );
   run_benchmarks_file< double >( benchmark, "s3dkt3m2.mtx" );
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-eigen-pi.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
#ifdef __CUDACC__
   config.addEntryEnum( "cuda" );
#endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) || ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String& logFileName = parameters.getParameter< String >( "log-file" );
   const String& outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   //const String devices = parameters.getParameter< String >( "devices" );
   run_benchmarks( benchmark );

   return EXIT_SUCCESS;
}
