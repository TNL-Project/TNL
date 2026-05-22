// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/BenchmarkResult.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

#include <type_traits>

template< typename PrecisionType >
struct EigenBenchmarkResult : TNL::Benchmarks::BenchmarkResult
{
   EigenBenchmarkResult( const PrecisionType& epsilon, const int& iterations, const double& error )
   : epsilon( epsilon ),
     iterations( iterations ),
     error( error )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.insert( headers.end(), { "epsilon", "iterations", "error" } );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << std::scientific << epsilon << ( iterations / loops ) << ( error / loops );
      return elements;
   }

   const PrecisionType& epsilon;
   const int& iterations;
   const double& error;
};

template< typename Device >
const char*
performer()
{
   if constexpr( std::is_same_v< Device, TNL::Devices::Host > )
#ifdef HAVE_OPENMP
      return "CPUP";
#else
      return "CPU";
#endif
   else if constexpr( std::is_same_v< Device, TNL::Devices::GPU > )
      return "GPU";
   else
      return "unknown";
}
