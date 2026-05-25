// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
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
#include <TNL/Solvers/Eigen/experimental/ShiftedPowerIteration.h>
#include <TNL/Solvers/Eigen/experimental/QRAlgorithm.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Algorithms/fillRandom.h>

#include <TNL/Benchmarks/Benchmark.h>
#include <string>
#include <type_traits>

#include "EigenBenchmarkResult.h"

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
   TNL::Algorithms::fillRandom< Device >(
      vector.getData(), vector.getSize(), static_cast< PrecisionType >( -1 ), static_cast< PrecisionType >( 1 ) );
   return vector;
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
void
benchmark_pi( Benchmark& benchmark, MatrixType& matrix, VectorType& initialVecOrig )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   Vector< double, Device > doubleEigenvector( matrix.getColumns() );
   doubleMatrix = matrix;
   constexpr int max_i = std::is_same_v< PrecisionType, float > ? 7 : 13;
   for( int i = 1; i <= max_i; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      benchmark.setMetadataElement( { "epsilon", TNL::convertToString( epsilon ) } );
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
            Solvers::Eigen::experimental::powerIteration< MatrixType >( matrix, epsilon, initialVec, 100000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( iterations, error );
      benchmark.time< Device >( resetFunction, "TNL", testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename MatrixType, typename VectorType, typename PrecisionType >
void
benchmark_spi( Benchmark& benchmark, MatrixType& matrix, VectorType& initialVecOrig, const PrecisionType& shiftValue )
{
   using DoubleMatrix = typename MatrixType::template Self< double >;
   PrecisionType eigenvalue = 0;
   DoubleMatrix doubleMatrix( matrix.getColumns(), matrix.getColumns() );
   doubleMatrix = matrix;
   constexpr int max_i = std::is_same_v< PrecisionType, float > ? 7 : 13;
   for( int i = 1; i <= max_i; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      benchmark.setMetadataElement( { "epsilon", TNL::convertToString( epsilon ) } );
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
            Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( matrix, epsilon, shiftValue, initialVec, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( iterations, error );
      benchmark.time< Device >( resetFunction, "TNL", testFunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO, typename VectorType >
void
run_benchmarks_DM(
   Benchmark& benchmark,
   const std::string& matrixName,
   int& size,
   MatrixTypeCMO& matrixCMO,
   VectorType& initialVecOrig,
   const PrecisionType& shiftValue )
{
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "operation", "PI" },
            { "precision", getType< PrecisionType >() },
            { "matrix name", matrixName },
            { "matrix type", "DM_CMO" },
            { "size", std::to_string( size ) },
         } ) );
   benchmark_pi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_CMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI0" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_CMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixTypeCMO, VectorType, PrecisionType >( benchmark, matrixCMO, initialVecOrig, 0 );
   }

   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "operation", "PI" },
            { "precision", getType< PrecisionType >() },
            { "matrix name", matrixName },
            { "matrix type", "DM_RMO" },
            { "size", std::to_string( size ) },
         } ) );
   benchmark_pi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_RMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI0" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "DM_RMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixTypeRMO, VectorType, PrecisionType >( benchmark, matrixRMO, initialVecOrig, 0 );
   }
}

template< typename Device, typename PrecisionType, typename MatrixType, typename VectorType >
void
run_benchmarks_SM(
   Benchmark& benchmark,
   const std::string& matrixName,
   const int& size,
   MatrixType& matrixSM,
   VectorType& initialVecOrig,
   const PrecisionType& shiftValue )
{
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "operation", "PI" },
            { "precision", getType< PrecisionType >() },
            { "matrix name", matrixName },
            { "matrix type", "SM" },
            { "size", std::to_string( size ) },
         } ) );
   benchmark_pi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig );
   if( shiftValue != 0 ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "SM" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig, shiftValue );
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "SPI0" },
               { "precision", getType< PrecisionType >() },
               { "matrix name", matrixName },
               { "matrix type", "SM" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_spi< Device, MatrixType, VectorType, PrecisionType >( benchmark, matrixSM, initialVecOrig, 0 );
   }
}

template< typename PrecisionType >
void
run_benchmarks_file( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& fileName = parameters.getParameter< std::string >( "input-matrix" );
   const auto& device = parameters.getParameter< std::string >( "device" );
   auto shiftValue = parameters.getParameter< double >( "shift-value" );

   std::string matrixName = fileName;
   matrixName.erase( matrixName.length() - 4, 4 );
   using MatrixTypeHostSM = Matrices::SparseMatrix< PrecisionType, Devices::Host, int >;
   MatrixTypeHostSM matrixSM;
   TNL::Matrices::MatrixReader< MatrixTypeHostSM >::readMtx( fileName, matrixSM );
   auto size = matrixSM.getColumns();
   using VectorTypeHost = Vector< PrecisionType, Devices::Host >;
   auto initialVecOrig = generateVector< VectorTypeHost >( size );

   if( device == "host" || device == "all" )
      run_benchmarks_SM< Devices::Host, PrecisionType >( benchmark, matrixName, size, matrixSM, initialVecOrig, shiftValue );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" ) {
      using VectorTypeGPU = Vector< PrecisionType, Devices::GPU >;
      VectorTypeGPU initialVecOrigGPU( size );
      initialVecOrigGPU = initialVecOrig;
      Matrices::SparseMatrix< PrecisionType, Devices::GPU, int > matrixGPUCMO( size, size );
      matrixGPUCMO = matrixSM;
      run_benchmarks_SM< Devices::GPU, PrecisionType >(
         benchmark, matrixName, size, matrixGPUCMO, initialVecOrigGPU, shiftValue );
   }
#endif

   if( size <= 1600 ) {
      using MatrixTypeHostCMO =
         Matrices::DenseMatrix< PrecisionType, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
      MatrixTypeHostCMO matrixDoubleDM;
      TNL::Matrices::MatrixReader< MatrixTypeHostCMO >::readMtx( fileName, matrixDoubleDM );

      if( device == "host" || device == "all" )
         run_benchmarks_DM< Devices::Host, PrecisionType >(
            benchmark, matrixName, size, matrixDoubleDM, initialVecOrig, shiftValue );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" ) {
         Matrices::DenseMatrix< PrecisionType, Devices::GPU, int, TNL::Algorithms::Segments::ColumnMajorOrder >
            matrixGPUDoubleCMO( size, size );
         matrixGPUDoubleCMO = matrixDoubleDM;
         using VectorTypeGPU = Vector< PrecisionType, Devices::GPU >;
         VectorTypeGPU initialVecOrigGPUDM( size );
         initialVecOrigGPUDM = initialVecOrig;
         run_benchmarks_DM< Devices::GPU, PrecisionType >(
            benchmark, matrixName, size, matrixGPUDoubleCMO, initialVecOrigGPUDM, shiftValue );
      }
#endif
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
   config.addEntry< double >( "shift-value", "Shift value for the shifted power iteration method.", 0.0 );
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
