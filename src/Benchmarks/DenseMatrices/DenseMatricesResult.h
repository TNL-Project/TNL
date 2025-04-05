// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>

namespace TNL::Benchmarks {

template< typename Real, typename Device, typename Index, typename ResultReal = Real, typename Logger = JsonLogging >
struct DenseMatricesResult : public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostMatrix = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;
   using BenchmarkMatrix = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;
   using BenchmarkResult::time_stddev;
   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;

   DenseMatricesResult( const HostMatrix& referenceResult, const std::vector< BenchmarkMatrix >& benchmarkResults )
   : referenceResult( referenceResult ),
     benchmarkResults( benchmarkResults )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = { "time", "time_stddev", "time_stddev/time", "loops", "bandwidth", "speedup" };
      for( size_t i = 0; i < benchmarkResults.size(); i++ ) {
         headers.push_back( "Diff.Max " + std::to_string( i + 1 ) );
         headers.push_back( "Diff.L2 " + std::to_string( i + 1 ) );
      }
      return headers;
   }

   [[nodiscard]] std::vector< IndexType >
   getColumnWidthHints() const override
   {
      std::vector< IndexType > columnWidths = { 14, 16, 18, 6, 14, 14 };
      for( size_t i = 0; i < benchmarkResults.size(); i++ ) {
         columnWidths.push_back( 14 );  // Width for "Diff.Max"
         columnWidths.push_back( 14 );  // Width for "Diff.L2"
      }
      return columnWidths;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements;
      elements << std::scientific << time << time_stddev << time_stddev / time << loops << bandwidth;

      if( speedup != 0.0 )
         elements << speedup;
      else
         elements << "N/A";

      // Compute and append differences for each benchmark result
      for( const auto& benchmarkMatrix : benchmarkResults ) {
         // Check if dimensions match
         auto diff = referenceResult.getValues() - benchmarkMatrix.getValues();
         elements << TNL::maxNorm( abs( diff ) ) << TNL::l2Norm( diff );
      }

      return elements;
   }

   const HostMatrix& referenceResult;
   std::vector< BenchmarkMatrix > benchmarkResults;  // Vector of benchmark matrices
};

}  // namespace TNL::Benchmarks
