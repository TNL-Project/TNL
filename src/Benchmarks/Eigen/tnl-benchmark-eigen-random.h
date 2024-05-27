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

template<typename VectorType >
VectorType
generateVector( int size )
{
   using PrecisionType = typename VectorType::RealType;
   using Device = typename VectorType::DeviceType;
   VectorType vector( size );
   TNL::Algorithms::fillRandom< Device >( vector.getData(), vector.getSize(), (PrecisionType) -1, (PrecisionType) 1 );
   return vector;
}

template< typename MatrixType >
MatrixType
generateMatrixDM( int size )
{
   using PrecisionType = typename MatrixType::RealType;
   using Device = typename MatrixType::DeviceType;
   MatrixType matrix( size, size );
   TNL::Algorithms::fillRandom< Device >( matrix.getValues().getData(), size * size, (PrecisionType) 0, (PrecisionType) 1 );
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
   // auto f = [ & ] __cuda_callable__( typename MatrixType::RowView & row )
   // {
   //    const int rowIdx = row.getRowIndex();
   //    const int capacity = row.getSize();
   //    for( int i = 0; i < capacity; i++ ) {
   //       int index = row.getColumnIndex( i );
   //       if( index > rowIdx ) {
   //          int indexNew = 0;
   //          for( int j = 0; j < i; j++ ) {
   //             int indexCurr = row.getColumnIndex( j );
   //             while( indexNew < indexCurr ) {
   //                if( indexCurr < indexNew ) {
   //                   row.setColumnIndex( i, indexNew );
   //                   break;
   //                }
   //                else
   //                   indexNew++;
   //             }
   //          }
   //       }
   //    }
   //    row.sortColumnIndexes();
   // };
   TNL::Containers::Vector< int, Device > rowCapacities( size );
   MatrixType matrix( size, size );
   TNL::Algorithms::fillRandom< Device >( rowCapacities.getData(), size, 1, size / 4 );
   rowCapacities.forAllElements(
      [] __cuda_callable__( int i, int& value )
      {
         if( value > i + 1 )
            value = i + 1;
      } );
   matrix.setRowCapacities( rowCapacities );
   int offset = 0;
   //std::cout << matrix.getColumnIndexes().getSize() << "\n";
   for( int i = 0; i < size; i++ ) {
      int value = rowCapacities[ i ];
      //std::cout << offset + value << "\n";
      TNL::Algorithms::fillRandom< Device >( matrix.getColumnIndexes().getData() + offset, value, 0, i );
      offset += value;
   }
   matrix.sortColumnIndexes();
   //matrix.forAllRows( f );
   TNL::Algorithms::fillRandom< Device >(
      matrix.getValues().getData(), matrix.getValues().getSize(), (PrecisionType) -1, (PrecisionType) 1 );
   return matrix;
}

template< typename Device, typename MatrixType, typename VectorType >
void
benchmark_pi( Benchmark<>& benchmark, MatrixType& matrix, VectorType& initialVecOrig )
{
   using PrecisionType = typename MatrixType::RealType;
   for( int i = 1; i < 15; i += 2 ) {
      PrecisionType epsilon = TNL::pow( 10.0, -i );
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
            Matrices::Eigen::powerIteration< PrecisionType, Device, MatrixType >( matrix, epsilon, initialVec, 10000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testfunction, eigenBenchmarkResult );
   }
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
      uint iter = 0;
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
            Matrices::Eigen::QRalgorithm< PrecisionType, Device, MatrixType >( matrix, epsilon, factorType, 5000 );
      };
      EigenBenchmarkResult eigenBenchmarkResult( epsilon, iterations, error );
      benchmark.time< Device >( resetFunction, performer< Device >(), testfunction, eigenBenchmarkResult );
      //std::cout << "Eigenvectors:\n" << eigenvectors << "\n";
      //std::cout << "Eigenvalues:\n" << eigenvalues << "\n";
   }
}

template< typename Device, typename PrecisionType, typename MatrixTypeCMO >
void
run_benchmarks_DM( Benchmark<>& benchmark, int size, MatrixTypeCMO& matrixCMO )
{
   using VectorType = Vector< PrecisionType, Device >;
   VectorType initialVecOrig = generateVector<VectorType>( matrixCMO.getColumns());
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_CMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixTypeCMO, VectorType >( benchmark, matrixCMO, initialVecOrig );

   using MatrixTypeRMO = Matrices::DenseMatrix< PrecisionType, Device, int, TNL::Algorithms::Segments::RowMajorOrder >;
   MatrixTypeRMO matrixRMO( size, size );
   matrixRMO = matrixCMO;
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "DM_RMO" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixTypeRMO, VectorType >( benchmark, matrixRMO, initialVecOrig );

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

template< typename Device, typename PrecisionType, typename MatrixType >
void
run_benchmarks_SM( Benchmark<>& benchmark, int size, MatrixType& matrixSM )
{
   using VectorType = Vector< PrecisionType, Device >;
   VectorType initialVecOrig = generateVector<VectorType>( matrixSM.getColumns());
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( {
      { "operation", "PI" },
      { "precision", getType< PrecisionType >() },
      { "matrixType", "SM" },
      { "size", std::to_string( size ) },
   } ) );
   benchmark_pi< Device, MatrixType, VectorType >( benchmark, matrixSM, initialVecOrig );
}

void
run_benchmarks( Benchmark<>& benchmark )
{
   using MatrixTypeHostFloatCMO =
      Matrices::DenseMatrix< float, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   using MatrixTypeHostDoubleCMO =
      Matrices::DenseMatrix< double, Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
   int size = 10;

      while( size <= 2000 ) {
         auto matrixHostFloatCMO = generateMatrixDM< MatrixTypeHostFloatCMO >( size );
         run_benchmarks_DM<Devices::Host, float >( benchmark, size, matrixHostFloatCMO );
         auto matrixHostDoubleCMO = generateMatrixDM< MatrixTypeHostDoubleCMO >( size );
         run_benchmarks_DM< Devices::Host, double >( benchmark, size, matrixHostDoubleCMO );
   #ifdef __CUDACC__
         //Matrices::DenseMatrix< float, Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder > matrixCUDAFloatCMO
         //=
         //matrixHostFloatCMO; run_benchmarks_DM< float >( benchmark, size, matrixCUDAFloatCMO );
         Matrices::DenseMatrix< double, Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >
         matrixCUDADoubleCMO(
            size, size );
         matrixCUDADoubleCMO = matrixHostDoubleCMO;
         run_benchmarks_DM< Devices::Cuda, double >( benchmark, size, matrixCUDADoubleCMO );
   #endif
         if( size == 10 || size == 200 ) {
            size *= 2.5;
         }
         else {
            size *= 2;
         }
         if( size > 2000 ) {
            break;
         }
      }
   using MatrixTypeHostFloatSM = Matrices::SparseMatrix< float, Devices::Host, int, Matrices::SymmetricMatrix >;
   using MatrixTypeHostDoubleSM = Matrices::SparseMatrix< double, Devices::Host, int, Matrices::SymmetricMatrix >;
   size = 100;
   while( size <= 10000 ) {
      //auto matrixHostFloatSM = generateMatrixSM< MatrixTypeHostFloatSM >( size );
      //run_benchmarks_SM< float >( benchmark, size, matrixHostFloatSM );
      auto matrixHostDoubleSM = generateMatrixSM< MatrixTypeHostDoubleSM >( size );
      run_benchmarks_SM< Devices::Host, double >( benchmark, size, matrixHostDoubleSM );
#ifdef __CUDACC__
      //Matrices::SparseMatrix< float, Devices::Cuda, int, Matrices::SymmetricMatrix > matrixCUDAFloatSM =
      //matrixHostFloatSM; run_benchmarks_DM< float >( benchmark, size, matrixCUDAFloatSM );
      Matrices::SparseMatrix< double, Devices::Cuda, int, Matrices::SymmetricMatrix > matrixCUDADoubleSM( size, size );
      matrixCUDADoubleSM = matrixHostDoubleSM;
      run_benchmarks_SM< Devices::Cuda, double >( benchmark, size, matrixCUDADoubleSM );
#endif
      if( size == 10 || size == 200 ) {
         size *= 2.5;
      }
      else if( size == 500 || size == 1000 || size == 5000 ) {
         size *= 1.5;
      }
      else if( size == 750 || size == 7500 ) {
         size += ( size / 3 );
      }
      else if( size == 1500 ) {
         size += 2 * ( size / 3 );
      }
      else {
         size *= 2;
      }
   }
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
