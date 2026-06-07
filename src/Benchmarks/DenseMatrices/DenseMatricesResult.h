// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>

namespace TNL::Benchmarks {

template< typename Real, typename Device, typename Index >
struct DenseMatricesResult : public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostMatrix = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;
   using BenchmarkMatrix = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;
   using Reference = std::pair< BenchmarkMatrix, std::string >;

   DenseMatricesResult( const HostMatrix& referenceResult, const std::vector< Reference >& references )
   : referenceResult( referenceResult ),
     references( references )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      for( const auto& ref : references ) {
         headers.push_back( "Diff.Max vs " + ref.second );
         headers.push_back( "Diff.L2 vs " + ref.second );
      }
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();

      // Compute and append differences for each reference result
      for( const auto& ref : references ) {
         auto diff = referenceResult.getValues() - ref.first.getValues();
         elements << TNL::maxNorm( abs( diff ) ) << TNL::l2Norm( diff );
      }

      return elements;
   }

   const HostMatrix& referenceResult;
   std::vector< Reference > references;
};

}  // namespace TNL::Benchmarks
