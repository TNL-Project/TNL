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

template< typename Device, typename MatrixType >
void
benchmark_qr( Benchmark<>& benchmark, MatrixType& matrix, Matrices::Factorization::QR::QRfactorizationType factorType )
{
   using PrecisionType = typename MatrixType::RealType;
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      PrecisionType error = 0;
      int iterations = 0;
      MatrixType eigenvalues( matrix.getColumns(), matrix.getColumns() );
      MatrixType eigenvectors( matrix.getColumns(), matrix.getColumns() );
      eigenvalues.setValue( 0 );
      eigenvectors.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         //std::cout << "Eigenvectors:\n" << eigenvectors << "\n";
         //std::cout << "Eigenvalues:\n" << eigenvalues << "\n";
         MatrixType matrixVector( matrix.getColumns(), matrix.getColumns() );
         matrixVector.getMatrixProduct( matrix, eigenvectors );
         auto f = [] __cuda_callable__( typename MatrixType::RowView & row )
         {
            const int& rowIdx = row.getRowIndex();
            int size = row.getSize();
            for( int i = 0; i < size; i++ )
               if( i != rowIdx )
                  row.setValue( i, 0 );
         };
         eigenvalues.forAllRows( f );
         //std::cout << "Eigenvalues:\n" << eigenvalues << "\n";
         //getchar();
         //std::cout << "MatrixVector:\n" << matrixVector << "\n";
         MatrixType valuesVector( matrix.getColumns(), matrix.getColumns() );
         valuesVector.getMatrixProduct( eigenvectors, eigenvalues );
         //std::cout << "valuesVector:\n" << valuesVector << "\n";
         matrixVector.addMatrix( valuesVector, -1 );
         //std::cout << "valuesVector:\n" << matrixVector << "\n";
         error += TNL::maxNorm( matrixVector.getValues() );
         iterations += iter;
      };
      auto testfunction = [ & ]()
      {
         std::tie( eigenvalues, eigenvectors, iter ) =
            Matrices::Eigen::QRalgorithm< PrecisionType, Device, MatrixType >( matrix, epsilon, factorType, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testfunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
      //std::cout << "Eigenvectors:\n" << eigenvectors << "\n";
      //std::cout << "Eigenvalues:\n" << eigenvalues << "\n";
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark<>& benchmark, int size, MatrixTypeCMO& matrixCMO )
{
   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;

   if( ! std::is_same_v< Device, Devices::Cuda > ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "operation", "QR" },
                                                                    { "precision", getType< PrecisionType >() },
                                                                    { "MatrixType", "DM_CMO" },
                                                                    { "size", std::to_string( size ) },
                                                                    { "facType", "HH" } } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::QRfactorizationType::HouseholderType );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "MatrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
         { "facType", "GM" },
      } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::QRfactorizationType::GramSchmidtType );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "MatrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
         { "facType", "GV" },
      } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::QRfactorizationType::GivensType );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "MatrixType", "DM_RMO" },
         { "size", std::to_string( size ) },
         { "facType", "GV" },
      } ) );
      benchmark_qr< Device, MatrixTypeRMO >(
         benchmark, matrixRMO, Matrices::Factorization::QR::QRfactorizationType::GivensType );
   }
}

void
run_banchmarks_file( Benchmark<>& benchmark, const std::string& fileName )
{
   using MatrixTypeHostFloatCMO =
      Matrices::DenseMatrix< float, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   using MatrixTypeHostDoubleCMO =
      Matrices::DenseMatrix< double, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   MatrixTypeHostDoubleCMO matrixDM;
   TNL::Matrices::MatrixReader< MatrixTypeHostDoubleCMO >::readMtx( fileName, matrixDM );
   int size = matrixDM.getColumns();
   run_benchmarks_DM< Devices::Host, double >( benchmark, size, matrixDM );
   MatrixTypeHostFloatCMO matrixFloatDM;
   TNL::Matrices::MatrixReader< MatrixTypeHostFloatCMO >::readMtx( fileName, matrixFloatDM );
   size = matrixDM.getColumns();
   run_benchmarks_DM< Devices::Host, float >( benchmark, size, matrixFloatDM );
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
