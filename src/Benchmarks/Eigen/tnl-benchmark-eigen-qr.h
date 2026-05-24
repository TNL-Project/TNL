// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/TypeInfo.h>
#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Solvers/Eigen/experimental/PowerIteration.h>
#include <TNL/Solvers/Eigen/experimental/QRAlgorithm.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Algorithms/fillRandom.h>

#include <TNL/Benchmarks/Benchmark.h>
#include <string>
#include <type_traits>

#include "EigenBenchmarkResult.h"

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Containers;

template< typename Device, typename MatrixType >
void
benchmark_qr( Benchmark& benchmark, MatrixType& matrix, const Matrices::Factorization::QR::FactorizationMethod& factorMethod )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   using PrecisionType = typename MatrixType::RealType;
   if constexpr( std::is_same< PrecisionType, double >() ) {
      doubleMatrix = matrix;
   }
   constexpr int max_i = std::is_same_v< PrecisionType, float > ? 7 : 13;
   for( int i = 1; i <= max_i; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      benchmark.setMetadataElement( { "epsilon", TNL::convertToString( epsilon ) } );
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
            Solvers::Eigen::experimental::QRAlgorithm< MatrixType >( matrix, epsilon, factorMethod, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark& benchmark, const std::string& matrixName, const int& size, MatrixTypeCMO& matrixCMO )
{
   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;

   if constexpr( ! std::is_same_v< Device, Devices::GPU > ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            { { "operation", "QR" },
              { "precision", getType< PrecisionType >() },
              { "matrix name", matrixName },
              { "matrix type", "DM_CMO" },
              { "size", std::to_string( size ) },
              { "factorization", "HH" } } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Householder );

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "QR" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_CMO" },
               { "size", std::to_string( size ) },
               { "factorization", "GM" },
            } ) );
      benchmark_qr< Device, MatrixTypeCMO >(
         benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "QR" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_CMO" },
               { "size", std::to_string( size ) },
               { "factorization", "GV" },
            } ) );
      benchmark_qr< Device, MatrixTypeCMO >( benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Givens );

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "QR" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_RMO" },
               { "size", std::to_string( size ) },
               { "factorization", "GV" },
            } ) );
      benchmark_qr< Device, MatrixTypeRMO >( benchmark, matrixRMO, Matrices::Factorization::QR::FactorizationMethod::Givens );
   }
}

template< typename PrecisionType >
void
run_benchmarks_file( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& fileName = parameters.getParameter< std::string >( "input-matrix" );
   const auto& device = parameters.getParameter< std::string >( "device" );

   std::string matrixName = fileName;
   matrixName.erase( matrixName.length() - 4, 4 );

   if( device == "host" || device == "all" ) {
      using MatrixTypeHostCMO =
         Matrices::DenseMatrix< PrecisionType, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
      MatrixTypeHostCMO matrixDM;
      TNL::Matrices::MatrixReader< MatrixTypeHostCMO >::readMtx( fileName, matrixDM );
      int size = matrixDM.getColumns();
      run_benchmarks_DM< Devices::Host, PrecisionType >( benchmark, matrixName, size, matrixDM );
   }
}

void
resolvePrecision( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      run_benchmarks_file< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      run_benchmarks_file< double >( benchmark, parameters );
}

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Eigen benchmark settings:" );
   config.addRequiredEntry< std::string >( "input-matrix", "Path to the input matrix in Matrix Market format (.mtx)." );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::GPU::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) || ! Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolvePrecision( benchmark, parameters );

   return EXIT_SUCCESS;
}
