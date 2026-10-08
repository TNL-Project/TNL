// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/BenchmarkResult.h>

struct EigenBenchmarkResult : TNL::Benchmarks::BenchmarkResult
{
   EigenBenchmarkResult( const int& iterations, const double& error )
   : iterations( iterations ),
     error( error )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.insert( headers.end(), { "iterations", "error" } );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << std::scientific << ( iterations / loops ) << ( error / loops );
      return elements;
   }

   const int& iterations;
   const double& error;
};
