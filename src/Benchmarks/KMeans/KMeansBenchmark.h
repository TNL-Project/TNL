// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Timer.h>
#include <TNL/TypeInfo.h>

#include "SparseKMeans.h"

namespace TNL::Benchmarks::KMeans {

/**
 * \brief Benchmark result extended with the time of one iteration and the inertia of the final centroids.
 */
struct KMeansResult : public BenchmarkResult
{
   KMeansResult( int iterations, std::function< double() > inertia )
   : iterations( iterations ),
     inertia( std::move( inertia ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "time per iteration" );
      headers.push_back( "inertia" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << time / iterations << inertia();
      return elements;
   }

   int iterations;
   std::function< double() > inertia;
};

/**
 * \brief Returns the initial centroids, the points with the indexes `i * n / K` for `i = 0, ..., K - 1`.
 *
 * The same initialization is used by the reference implementations in `kmeans-reference.py`.
 */
template< typename Real, typename Index >
Matrices::DenseMatrix< Real, Devices::Host, Index >
getInitialCentroids( const Matrices::SparseMatrix< Real, Devices::Host, Index >& X, Index clusters )
{
   Matrices::DenseMatrix< Real, Devices::Host, Index > centroids( clusters, X.getColumns() );
   centroids.setValue( 0 );
   for( Index cluster = 0; cluster < clusters; cluster++ ) {
      const auto row = X.getRow( std::int64_t( cluster ) * X.getRows() / clusters );
      for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ )
         if( row.getColumnIndex( localIdx ) >= 0 )
            centroids.setElement( cluster, row.getColumnIndex( localIdx ), row.getValue( localIdx ) );
   }
   return centroids;
}

template< typename Device, typename Real, typename Index >
void
benchmarkDevice(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostX,
   Index clusters )
{
   const int iterations = parameters.getParameter< int >( "iterations" );
   const auto assignments = parameters.getList< std::string >( "assignment" );
   const auto updates = parameters.getList< std::string >( "update" );
   const auto reorderingPeriods = parameters.getList< int >( "reordering-period" );
   auto isSelected = []( const std::vector< std::string >& list, const std::string& item )
   {
      return std::find( list.begin(), list.end(), "all" ) != list.end()
          || std::find( list.begin(), list.end(), item ) != list.end();
   };

   Matrices::SparseMatrix< Real, Device, Index > X;
   X = hostX;
   const auto hostCentroids = getInitialCentroids( hostX, clusters );
   typename SparseKMeans< Real, Device, Index >::DenseMatrixType initialCentroids;
   initialCentroids = hostCentroids;
   SparseKMeans< Real, Device, Index > kmeans( X, clusters );
   // the tile of the clusters of the assignment is selected by measuring the assignment before the benchmark
   kmeans.setCentroids( initialCentroids );
   Index tileSize = parameters.getParameter< int >( "assignment-tile" );
   Timer tuningTimer;
   tuningTimer.start();
   if( tileSize > 0 )
      kmeans.setAssignmentTileSize( tileSize );
   else
      tileSize = kmeans.tuneAssignment();
   tuningTimer.stop();

   for( const auto& [ assignment, assignmentName ] : { std::pair{ Assignment::Fused, "fused" },
                                                       std::pair{ Assignment::FusedLambda, "fused-lambda" },
                                                       std::pair{ Assignment::Unfused, "unfused" } } )
   {
      if( ! isSelected( assignments, assignmentName ) )
         continue;
      for( const auto& [ update, updateName ] : { std::pair{ Update::SpMM, "spmm" },
                                                  std::pair{ Update::Atomic, "atomic" },
                                                  std::pair{ Update::Histogram, "histogram" } } )
      {
         if( ! isSelected( updates, updateName ) )
            continue;
         for( const int period : reorderingPeriods ) {
            // structured bindings cannot be captured in C++17
            const Assignment assignmentMethod = assignment;
            const Update updateMethod = update;
            auto compute = [ & ]()
            {
               kmeans.setCentroids( initialCentroids );
               kmeans.setReorderingPeriod( period );
               for( int i = 0; i < iterations; i++ )
                  kmeans.iterate( assignmentMethod, updateMethod );
            };
            // the inertia of the final centroids, like in scikit-learn
            KMeansResult result(
               iterations,
               [ & ]()
               {
                  kmeans.assignFused();
                  return kmeans.getInertia();
               } );
            benchmark.setMetadataElement( { "assignment", assignmentName } );
            benchmark.setMetadataElement( { "update", updateName } );
            benchmark.setMetadataElement( { "reordering", std::to_string( period ) } );
            benchmark.setMetadataElement( { "assignment tile", std::to_string( tileSize ) } );
            benchmark.setMetadataElement( { "tuning time", std::to_string( tuningTimer.getRealTime() ) } );
            benchmark.time< Device >( "TNL", compute, result );
         }
      }
   }
}

template< typename Real, typename Index >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   using HostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index >;
   const auto& inputFile = parameters.getParameter< std::string >( "input-file" );
   const auto& device = parameters.getParameter< std::string >( "device" );
   const int iterations = parameters.getParameter< int >( "iterations" );
   HostMatrix X;
   Matrices::MatrixReader< HostMatrix >::readMtx( inputFile, X );
   const Index nonzeros = X.getNonzeroElementsCount();

   for( const int clusters : parameters.getList< int >( "clusters" ) ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "matrix", inputFile },
               { "points", std::to_string( X.getRows() ) },
               { "features", std::to_string( X.getColumns() ) },
               { "nonzeros", std::to_string( nonzeros ) },
               { "precision", getType< Real >() },
               { "clusters", std::to_string( clusters ) },
               { "iterations", std::to_string( iterations ) },
            } ) );
      benchmark.setOperation( "k-means" );
      // the minimal memory traffic of one iteration: X twice (assignment and update) and the centroids twice
      benchmark.setDatasetSize(
         iterations
         * ( 2 * nonzeros * ( sizeof( Real ) + sizeof( Index ) )
             + 2 * std::size_t( X.getColumns() ) * clusters * sizeof( Real ) ) );
      // the dot products of the assignment and the sums of the update
      benchmark.setOperationsPerLoop( iterations * 4 * std::size_t( nonzeros ) * clusters );

      if( device == "host" || device == "all" )
         benchmarkDevice< Devices::Host >( benchmark, parameters, X, Index( clusters ) );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" )
         benchmarkDevice< Devices::GPU >( benchmark, parameters, X, Index( clusters ) );
#endif
   }
}

}  // namespace TNL::Benchmarks::KMeans
