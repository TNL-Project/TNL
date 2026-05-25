// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/MatrixType.h>
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
#include <algorithm>
#include <type_traits>

#include "EigenBenchmarkResult.h"

using namespace TNL;
using namespace Benchmarks;
using namespace Containers;

template< typename VectorType >
VectorType
generateVector( int size )
{
   using PrecisionType = typename VectorType::RealType;
   using Device = typename VectorType::DeviceType;
   VectorType vector( size );
   TNL::Algorithms::fillRandom< Device >(
      vector.getData(), vector.getSize(), static_cast< PrecisionType >( -1 ), static_cast< PrecisionType >( 1 ) );
   return vector;
}

template< typename MatrixType >
MatrixType
generateMatrixDM( int size )
{
   using PrecisionType = typename MatrixType::RealType;
   using Device = typename MatrixType::DeviceType;
   MatrixType matrix( size, size );
   TNL::Algorithms::fillRandom< Device >(
      matrix.getValues().getData(), size * size, static_cast< PrecisionType >( 0 ), static_cast< PrecisionType >( 1 ) );
   for( int i = 0; i < size; ++i ) {
      for( int j = 0; j < i; ++j ) {
         matrix.setElement( i, j, matrix.getElement( j, i ) );
      }
   }
   return matrix;
}

template< typename MatrixType >
MatrixType
generateMatrixSM( int size )
{
   using PrecisionType = typename MatrixType::RealType;
   using Device = typename MatrixType::DeviceType;
   TNL::Containers::Vector< int, Device > rowCapacities( size );
   MatrixType matrix( size, size );
   TNL::Algorithms::fillRandom< Device >( rowCapacities.getData(), size, 1, size / 4 );
   rowCapacities.forAllElements(
      [] __cuda_callable__( int i, int& value )
      {
         value = std::min( value, i + 1 );
      } );
   matrix.setRowCapacities( rowCapacities );
   int offset = 0;
   for( int i = 0; i < size; i++ ) {
      int value = rowCapacities[ i ];
      TNL::Algorithms::fillRandom< Device >( matrix.getColumnIndexes().getData() + offset, value, 0, i );
      offset += value;
   }
   matrix.sortColumnIndexes();
   //matrix.forAllRows( f );
   TNL::Algorithms::fillRandom< Device >(
      matrix.getValues().getData(),
      matrix.getValues().getSize(),
      static_cast< PrecisionType >( -1 ),
      static_cast< PrecisionType >( 1 ) );
   return matrix;
}

template< typename Device, typename MatrixType, typename VectorType >
void
benchmark_pi( Benchmark& benchmark, MatrixType& matrix, VectorType& initialVecOrig )
{
   using PrecisionType = typename MatrixType::RealType;
   constexpr int max_i = std::is_same_v< PrecisionType, float > ? 7 : 13;
   for( int i = 1; i <= max_i; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      benchmark.setMetadataElement( { "epsilon", TNL::convertToString( epsilon ) } );
      PrecisionType error = 0;
      int iterations = 0;
      PrecisionType eigenvalue = 0;
      VectorType eigenvector( matrix.getColumns() );
      VectorType initialVec( matrix.getColumns() );
      eigenvector.setValue( 0 );
      initialVec.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
         Vector< PrecisionType, Device > matrixEigenvector( matrix.getColumns() );
         matrix.vectorProduct( eigenvector, matrixEigenvector );
         error += TNL::maxNorm( ( eigenvalue * eigenvector ) - matrixEigenvector );
         iterations += iter;
         initialVec = initialVecOrig;
      };
      auto testfunction = [ & ]()
      {
         std::tie( eigenvalue, eigenvector, iter ) =
            Solvers::Eigen::experimental::powerIteration< MatrixType >( matrix, epsilon, initialVec, 100000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( iterations, error );
      benchmark.time< Device >( resetFunction, getDeviceName< Device >(), testfunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename MatrixType >
void
benchmark_qr( Benchmark& benchmark, MatrixType& matrix, Matrices::Factorization::QR::FactorizationMethod factorType )
{
   using PrecisionType = typename MatrixType::RealType;
   constexpr int max_i = std::is_same_v< PrecisionType, float > ? 7 : 13;
   for( int i = 1; i <= max_i; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
      benchmark.setMetadataElement( { "epsilon", TNL::convertToString( epsilon ) } );
      PrecisionType error = 0;
      int iterations = 0;
      MatrixType eigenvalues( matrix.getColumns(), matrix.getColumns() );
      MatrixType eigenvectors( matrix.getColumns(), matrix.getColumns() );
      eigenvalues.setValue( 0 );
      eigenvectors.setValue( 0 );
      int iter = 0;
      auto resetFunction = [ & ]()
      {
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
         valuesVector.getMatrixProduct( eigenvectors, eigenvalues );
         matrixVector.addMatrix( valuesVector, -1 );
         error += TNL::maxNorm( matrixVector.getValues() );
         iterations += iter;
      };
      auto testfunction = [ & ]()
      {
         std::tie( eigenvalues, eigenvectors, iter ) =
            Solvers::Eigen::experimental::QRAlgorithm< MatrixType >( matrix, epsilon, factorType, 5000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( iterations, error );
      benchmark.time< Device >( resetFunction, getDeviceName< Device >(), testfunction, eigenBenchmarkResult );
      if( iterations == 0 )
         break;
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters, MatrixTypeCMO& matrixCMO )
{
   const bool withPI = parameters.getParameter< bool >( "with-pi" );
   const bool withQRHouseholder = parameters.getParameter< bool >( "with-qr-householder" );
   const bool withQRGramSchmidt = parameters.getParameter< bool >( "with-qr-gram-schmidt" );
   const bool withQRGivens = parameters.getParameter< bool >( "with-qr-givens" );
   const int size = matrixCMO.getColumns();

   using VectorType = Vector< PrecisionType, Device >;
   auto initialVecOrig = generateVector< VectorType >( matrixCMO.getColumns() );

   if( withPI ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "PI" },
               { "precision", getType< PrecisionType >() },
               { "matrix type", "DM_CMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_pi< Device >( benchmark, matrixCMO, initialVecOrig );

      using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
      MatrixTypeRMO matrixRMO( size, size );
      matrixRMO = matrixCMO;
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "operation", "PI" },
               { "precision", getType< PrecisionType >() },
               { "matrix type", "DM_RMO" },
               { "size", std::to_string( size ) },
            } ) );
      benchmark_pi< Device >( benchmark, matrixRMO, initialVecOrig );
   }

   if constexpr( ! std::is_same_v< Device, Devices::GPU > ) {
      if( withQRHouseholder ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               { { "operation", "QR" },
                 { "precision", getType< PrecisionType >() },
                 { "matrix type", "DM_CMO" },
                 { "size", std::to_string( size ) },
                 { "factorization", "Householder" } } ) );
         benchmark_qr< Device, MatrixTypeCMO >(
            benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Householder );
      }

      if( withQRGramSchmidt ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "operation", "QR" },
                  { "precision", getType< PrecisionType >() },
                  { "matrix type", "DM_CMO" },
                  { "size", std::to_string( size ) },
                  { "factorization", "GramSchmidt" },
               } ) );
         benchmark_qr< Device >( benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
      }

      if( withQRGivens ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "operation", "QR" },
                  { "precision", getType< PrecisionType >() },
                  { "matrix type", "DM_CMO" },
                  { "size", std::to_string( size ) },
                  { "factorization", "Givens" },
               } ) );
         benchmark_qr< Device >( benchmark, matrixCMO, Matrices::Factorization::QR::FactorizationMethod::Givens );

         using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
         MatrixTypeRMO matrixRMO( size, size );
         matrixRMO = matrixCMO;
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "operation", "QR" },
                  { "precision", getType< PrecisionType >() },
                  { "matrix type", "DM_RMO" },
                  { "size", std::to_string( size ) },
                  { "factorization", "Givens" },
               } ) );
         benchmark_qr< Device >( benchmark, matrixRMO, Matrices::Factorization::QR::FactorizationMethod::Givens );
      }
   }
}

template< typename Device, typename PrecisionType, typename MatrixType >
void
run_benchmarks_SM(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   MatrixType& matrixSM )
{
   const bool withPI = parameters.getParameter< bool >( "with-pi" );
   if( ! withPI )
      return;
   const int size = matrixSM.getColumns();

   using VectorType = TNL::Containers::Vector< PrecisionType, Device >;
   auto initialVecOrig = generateVector< VectorType >( matrixSM.getColumns() );
   benchmark.setMetadataColumns(
      TNL::Benchmarks::Benchmark::MetadataColumns(
         {
            { "operation", "PI" },
            { "precision", TNL::getType< PrecisionType >() },
            { "matrix type", "SM" },
            { "size", std::to_string( size ) },
         } ) );
   benchmark_pi< Device >( benchmark, matrixSM, initialVecOrig );
}

template< typename PrecisionType >
void
run_benchmarks( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   const int minSizeDense = parameters.getParameter< int >( "min-size-dense" );
   const int maxSizeDense = parameters.getParameter< int >( "max-size-dense" );
   const int minSizeSparse = parameters.getParameter< int >( "min-size-sparse" );
   const int maxSizeSparse = parameters.getParameter< int >( "max-size-sparse" );

   using MatrixTypeHostCMO =
      TNL::Matrices::DenseMatrix< PrecisionType, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   for( int size = minSizeDense; size <= maxSizeDense; size *= 2 ) {
      if( device == "host" || device == "all" ) {
         auto matrixHostCMO = generateMatrixDM< MatrixTypeHostCMO >( size );
         run_benchmarks_DM< TNL::Devices::Host, PrecisionType >( benchmark, parameters, matrixHostCMO );
      }
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" ) {
         auto matrixHostCMO = generateMatrixDM< MatrixTypeHostCMO >( size );
         TNL::Matrices::DenseMatrix< PrecisionType, TNL::Devices::GPU, int, TNL::Algorithms::Segments::ColumnMajorOrder >
            matrixGPUCMO( size, size );
         matrixGPUCMO = matrixHostCMO;
         run_benchmarks_DM< TNL::Devices::GPU, PrecisionType >( benchmark, parameters, matrixGPUCMO );
      }
#endif
   }

   using MatrixTypeHostSM =
      TNL::Matrices::SparseMatrix< PrecisionType, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >;
   for( int size = minSizeSparse; size <= maxSizeSparse; size *= 2 ) {
      if( device == "host" || device == "all" ) {
         auto matrixHostSM = generateMatrixSM< MatrixTypeHostSM >( size );
         run_benchmarks_SM< TNL::Devices::Host, PrecisionType >( benchmark, parameters, matrixHostSM );
      }
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" ) {
         auto matrixHostSM = generateMatrixSM< MatrixTypeHostSM >( size );
         TNL::Matrices::SparseMatrix< PrecisionType, TNL::Devices::GPU, int, TNL::Matrices::SymmetricMatrix > matrixGPUSM(
            size, size );
         matrixGPUSM = matrixHostSM;
         run_benchmarks_SM< TNL::Devices::GPU, PrecisionType >( benchmark, parameters, matrixGPUSM );
      }
#endif
   }
}

void
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      run_benchmarks< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      run_benchmarks< double >( benchmark, parameters );
}

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Eigen benchmark settings:" );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "all" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "min-size-dense", "Minimum dense matrix size.", 10 );
   config.addEntry< int >( "max-size-dense", "Maximum dense matrix size.", 2000 );
   config.addEntry< int >( "min-size-sparse", "Minimum sparse matrix size.", 100 );
   config.addEntry< int >( "max-size-sparse", "Maximum sparse matrix size.", 10000 );
   config.addEntry< bool >( "with-pi", "Run power iteration benchmarks.", true );
   config.addEntry< bool >( "with-qr-householder", "Run QR algorithm with Householder factorization.", true );
   config.addEntry< bool >( "with-qr-gram-schmidt", "Run QR algorithm with Gram-Schmidt factorization.", true );
   config.addEntry< bool >( "with-qr-givens", "Run QR algorithm with Givens factorization.", true );

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
