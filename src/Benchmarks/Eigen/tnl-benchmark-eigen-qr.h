// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/TypeInfo.h>
#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Solvers/Eigenvalues/experimental/PowerIteration.h>
#include <TNL/Solvers/Eigenvalues/experimental/QRAlgorithm.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Algorithms/fillRandom.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

#include "EigenBenchmark.h"

using namespace TNL;
using namespace Benchmarks;
using namespace Containers;

template< typename Device, typename MatrixType >
void
benchmark_qr( Benchmark<>& benchmark, MatrixType& matrix, const Matrices::Factorization::QR::FactorizationMethod& factorMethod )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   using PrecisionType = typename MatrixType::RealType;
   if constexpr( std::is_same< PrecisionType, double >() ) {
      doubleMatrix = matrix;
   }
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      double error = 0;
      int iterations = 0;
      MatrixType eigenvalues( matrix.getColumns(), matrix.getColumns() );
      MatrixType eigenvectors( matrix.getColumns(), matrix.getColumns() );
      eigenvalues.setValue( 0 );
      eigenvectors.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         if constexpr( std::is_same< PrecisionType, double >() ) {
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
            MatrixType valuesVector( matrix.getColumns(), matrix.getColumns() );
            valuesVector.getMatrixProduct( eigenvalues, eigenvectors );
            matrixVector.addMatrix( valuesVector, -1 );
            error += TNL::maxNorm( matrixVector.getValues() );
         }
         else {
            DoubleMatrix matrixVector( matrix.getColumns(), matrix.getColumns() );
            DoubleMatrix doubleEigenvalues( matrix.getColumns(), matrix.getColumns() );
            DoubleMatrix doubleEigenvectors( matrix.getColumns(), matrix.getColumns() );
            doubleEigenvalues = eigenvalues;
            doubleEigenvectors = eigenvectors;
            matrixVector.getMatrixProduct( doubleMatrix, doubleEigenvectors );
            auto f = [] __cuda_callable__( typename DoubleMatrix::RowView & row )
            {
               const int& rowIdx = row.getRowIndex();
               int size = row.getSize();
               for( int i = 0; i < size; i++ )
                  if( i != rowIdx )
                     row.setValue( i, 0 );
            };
            doubleEigenvalues.forAllRows( f );
            DoubleMatrix valuesVector( matrix.getColumns(), matrix.getColumns() );
            valuesVector.getMatrixProduct( doubleEigenvalues, doubleEigenvectors );
            matrixVector.addMatrix( valuesVector, -1 );
            error += TNL::maxNorm( matrixVector.getValues() );
         }
         iterations += iter;
      };
      auto testFunction = [ & ]()
      {
         std::tie( eigenvalues, eigenvectors, iter ) =
            Solvers::Eigenvalues::experimental::QRAlgorithm< MatrixType >( matrix, epsilon, factorMethod, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark<>& benchmark, const std::string& matrixName, const int& size, MatrixTypeCMO& matrixCMO )
{
   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;

   if( ! std::is_same_v< Device, Devices::Cuda > ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "operation", "QR" },
                                                                    { "precision", getType< PrecisionType >() },
                                                                    { "matrixName", matrixName },
                                                                    { "matrixType", "DM_CMO" },
                                                                    { "size", std::to_string( size ) },
                                                                    { "facMethod", "HH" } } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Householder );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
         { "facMethod", "GM" },
      } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_CMO" },
         { "size", std::to_string( size ) },
         { "facMethod", "GV" },
      } ) );
      benchmark_qr< Device, MatrixTypeCMO >( benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Givens );

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
         { "operation", "QR" },
         { "precision", getType< PrecisionType >() },
         { "matrixName", matrixName },
         { "matrixType", "DM_RMO" },
         { "size", std::to_string( size ) },
         { "facMethod", "GV" },
      } ) );
      benchmark_qr< Device, MatrixTypeRMO >( benchmark, matrixRMO, Matrices::Factorization::QR::FactorizationMethod::Givens );
   }
}

void
run_benchmarks_file( Benchmark<>& benchmark, const std::string& fileName )
{
   std::string matrixName = fileName;
   matrixName.erase( matrixName.length() - 4, 4 );
   using MatrixTypeHostFloatCMO =
      Matrices::DenseMatrix< float, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   using MatrixTypeHostDoubleCMO =
      Matrices::DenseMatrix< double, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   MatrixTypeHostDoubleCMO matrixDM;
   TNL::Matrices::MatrixReader< MatrixTypeHostDoubleCMO >::readMtx( fileName, matrixDM );
   int size = matrixDM.getColumns();
   run_benchmarks_DM< Devices::Host, double >( benchmark, matrixName, size, matrixDM );
   MatrixTypeHostFloatCMO matrixFloatDM;
   TNL::Matrices::MatrixReader< MatrixTypeHostFloatCMO >::readMtx( fileName, matrixFloatDM );
   size = matrixDM.getColumns();
   run_benchmarks_DM< Devices::Host, float >( benchmark, matrixName, size, matrixFloatDM );
}

void
run_benchmarks( Benchmark<>& benchmark )
{
   //https://sparse.tamu.edu/HB/bcspwr01
   run_benchmarks_file( benchmark, "bcspwr01.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk01
   run_benchmarks_file( benchmark, "bcsstk01.mtx" );
   //http://sparse.tamu.edu/Bai/bfwb62
   run_benchmarks_file( benchmark, "bfwb62.mtx" );
   //https://sparse.tamu.edu/Newman/polbooks
   run_benchmarks_file( benchmark, "polbooks.mtx" );
   //https://sparse.tamu.edu/HB/bcsstm03
   run_benchmarks_file( benchmark, "bcsstm03.mtx" );
   //https://sparse.tamu.edu/HB/bcsstm05
   run_benchmarks_file( benchmark, "bcsstm05.mtx" );
   //https://sparse.tamu.edu/HB/bcsstk05
   run_benchmarks_file( benchmark, "bcsstk05.mtx" );
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-eigen-qr.log" );
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
