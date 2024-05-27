// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "TNL/Algorithms/Segments/ElementsOrganization.h"
#include "TNL/Containers/Vector.h"
#include "TNL/Matrices/MatrixType.h"
#include "TNL/TypeInfo.h"
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
//#include <iostream>
//#include <ostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace TNL;
using namespace Benchmarks;
using namespace Containers;

template< typename PrecisionType >
struct EigenBenchmarkResult : BenchmarkResult
{
   EigenBenchmarkResult( const PrecisionType& epsilon, const int& iterations, const PrecisionType& error )
   : epsilon( epsilon ), iterations( iterations ), error( error )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      return HeaderElements( { "time", "stddev", "stddev/time", "loops", "epsilon", "iterations", "error" } );
   }

   [[nodiscard]] std::vector< int >
   getColumnWidthHints() const override
   {
      return std::vector< int >( { 14, 14, 14, 6, 14, 12, 14 } );
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time << time_stddev << time_stddev / time << loops << epsilon << ( iterations / loops )
               << ( error / loops );
      return elements;
   }

   const PrecisionType& epsilon;
   const int& iterations;
   const PrecisionType& error;
};

template< typename Device >
const char*
performer()
{
   if( std::is_same< Device, Devices::Host >::value )
      return "CPU";
   else if( std::is_same< Device, Devices::Cuda >::value )
      return "GPU";
   else
      return "unknown";
}

template< typename VectorType >
VectorType
generateVector( int size )
{
   using PrecisionType = typename VectorType::RealType;
   using Device = typename VectorType::DeviceType;
   VectorType vector( size );
   TNL::Algorithms::fillRandom< Device >( vector.getData(), vector.getSize(), (PrecisionType) -1, (PrecisionType) 1 );
   return vector;
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
PrecisionType
benchmark_pi( Benchmark<>& benchmark, MatrixType& matrix, VectorType& initialVecOrig )
{
   PrecisionType eigenvalue = 0;
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      PrecisionType error = 0;
      int iterations = 0;
      VectorType eigenvector( matrix.getColumns() );
      VectorType initialVec( matrix.getColumns() );
      eigenvector.setValue( 0 );
      initialVec.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         //std::cout << eigenvector;
         //std::cout << eigenvalue << "\n";
         Vector< PrecisionType, Device > matrixEigenvector( matrix.getColumns() );
         matrix.vectorProduct( eigenvector, matrixEigenvector );
         error += TNL::maxNorm( ( eigenvalue * eigenvector ) - matrixEigenvector );
         iterations += iter;
         initialVec = initialVecOrig;
      };
      auto testfunction = [ & ]()
      {
         std::tie( eigenvalue, eigenvector, iter ) =
            Matrices::Eigen::powerIteration< PrecisionType, Device, MatrixType >( matrix, epsilon, initialVec, 100000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testfunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
   return eigenvalue;
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
void
benchmark_spi( Benchmark<>& benchmark, MatrixType& matrix, VectorType& initialVecOrig, PrecisionType shiftValue )
{
   PrecisionType eigenvalue = 0;
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      PrecisionType error = 0;
      int iterations = 0;
      VectorType eigenvector( matrix.getColumns() );
      VectorType initialVec( matrix.getColumns() );
      eigenvector.setValue( 0 );
      initialVec.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         //std::cout << eigenvector;
         //std::cout << eigenvalue << "\n";
         Vector< PrecisionType, Device > matrixEigenvector( matrix.getColumns() );
         matrix.vectorProduct( eigenvector, matrixEigenvector );
         error += TNL::maxNorm( ( eigenvalue * eigenvector ) - matrixEigenvector );
         iterations += iter;
         initialVec = initialVecOrig;
      };
      auto testfunction = [ & ]()
      {
         std::tie( eigenvalue, eigenvector, iter ) =
            Matrices::Eigen::shiftedPowerIteration< PrecisionType, Device, MatrixType >( matrix, epsilon, shiftValue, initialVec, 100000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testfunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark<>& benchmark, int size, MatrixTypeCMO& matrixCMO )
{
   using VectorType = Vector< PrecisionType, Device >;
   VectorType initialVecOrig = generateVector< VectorType >( matrixCMO.getColumns() );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_CMO" },
      { "size", std::to_string( size ) },
   } ) );
   PrecisionType eigenvalue = benchmark_pi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "SPI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_CMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_spi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig, eigenvalue );

   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_RMO" },
      { "size", std::to_string( size ) },
   } ) );
   eigenvalue = benchmark_pi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "SPI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_RMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_spi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig, eigenvalue );
}

template< typename Device, typename PrecisionType, typename MatrixType >
void
run_benchmarks_SM( Benchmark<>& benchmark, int size, MatrixType& matrixSM )
{
   using VectorType = Vector< PrecisionType, Device >;
   VectorType initialVecOrig = generateVector< VectorType >( matrixSM.getColumns() );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "SM" },
      { "size", std::to_string( size ) },
   } ) );
   PrecisionType eigenvalue = benchmark_pi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig );
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "SPI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "SM" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_spi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig, eigenvalue );
}

void
run_banchmarks_file( Benchmark<>& benchmark, const std::string& fileName )
{
   using MatrixTypeHostFloatSM = Matrices::SparseMatrix< float, Devices::Host, int >;
   using MatrixTypeHostDoubleSM = Matrices::SparseMatrix< double, Devices::Host, int >;
   MatrixTypeHostDoubleSM matrixSM;
   TNL::Matrices::MatrixReader< MatrixTypeHostDoubleSM >::readMtx( fileName, matrixSM );
   int size = matrixSM.getColumns();
   run_benchmarks_SM< Devices::Host, double >( benchmark, size, matrixSM );
   MatrixTypeHostFloatSM matrixFloatSM;
   TNL::Matrices::MatrixReader< MatrixTypeHostFloatSM >::readMtx( fileName, matrixFloatSM );
   run_benchmarks_SM< Devices::Host, float >( benchmark, size, matrixFloatSM );
#ifdef __CUDACC__
   Matrices::SparseMatrix< double, Devices::Cuda, int > matrixCUDADoubleCMO( size, size );
   matrixCUDADoubleCMO = matrixFloatSM;
   run_benchmarks_SM< Devices::Cuda, double >( benchmark, size, matrixCUDADoubleCMO );
   Matrices::SparseMatrix< float, Devices::Cuda, int > matrixCUDAFloatCMO( size, size );
   matrixCUDAFloatCMO = matrixFloatSM;
   run_benchmarks_SM< Devices::Cuda, float >( benchmark, size, matrixCUDAFloatCMO );
#endif
   if( size <= 1200 ) {
      using MatrixTypeHostFloatCMO =
         Matrices::DenseMatrix< float, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
      using MatrixTypeHostDoubleCMO =
         Matrices::DenseMatrix< double, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
      MatrixTypeHostDoubleCMO matrixDoubleDM;
      TNL::Matrices::MatrixReader< MatrixTypeHostDoubleCMO >::readMtx( fileName, matrixDoubleDM );
      run_benchmarks_DM< Devices::Host, double >( benchmark, size, matrixDoubleDM );
      MatrixTypeHostFloatCMO matrixFloatDM;
      TNL::Matrices::MatrixReader< MatrixTypeHostFloatCMO >::readMtx( fileName, matrixFloatDM );
      run_benchmarks_DM< Devices::Host, float >( benchmark, size, matrixFloatDM );
#ifdef __CUDACC__
      Matrices::DenseMatrix< double, Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder > matrixCUDADoubleCMO(
         size, size );
      matrixCUDADoubleCMO = matrixDoubleDM;
      run_benchmarks_DM< Devices::Cuda, double >( benchmark, size, matrixCUDADoubleCMO );
      Matrices::DenseMatrix< float, Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder > matrixCUDAFloatCMO(
         size, size );
      matrixCUDAFloatCMO = matrixFloatDM;
      run_benchmarks_DM< Devices::Cuda, float >( benchmark, size, matrixCUDAFloatCMO );
#endif
   }
}

void
run_benchmarks( Benchmark<>& benchmark )
{
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcspwr01.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk01.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bfw62b.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstm03.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstm05.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk05.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/polbooks.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/plat362.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bfw782a.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/lshp1561.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk13.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk21.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk29.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk30.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/bcsstk32.mtx" );
   run_banchmarks_file( benchmark, "/home/salabmar/tnl/.test_matrices/s3dkt3m2.mtx" );
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-eigen.log" );
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
