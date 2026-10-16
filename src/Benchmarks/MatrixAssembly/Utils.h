// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <exception>
#include <string>

#include <TNL/Benchmarks/Benchmark.h>

namespace TNL::Benchmarks::MatrixAssembly {

//! \brief Reset function which does nothing.
inline void
noReset()
{}

/**
 * \brief Sets the operation and measures the compute function.
 *
 * Exceptions thrown from the reset or compute functions (e.g. failed allocations)
 * are reported as error messages and do not stop the benchmark.
 */
template< typename Device, typename ResetFunction, typename ComputeFunction >
void
timeOperation(
   Benchmark& benchmark,
   const std::string& operation,
   const std::string& performer,
   ResetFunction&& reset,
   ComputeFunction& compute )
{
   benchmark.setOperation( operation );
   try {
      benchmark.time< Device >( reset, performer, compute );
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( performer + " " + operation + " failed: " + e.what() );
   }
}

//! \brief Reports the result of the verification of the assembled matrix.
inline void
reportVerification( Benchmark& benchmark, const std::string& performer, const std::string& error )
{
   if( ! error.empty() )
      benchmark.addErrorMessage( performer + ": verification of the assembled matrix failed: " + error );
}

}  // namespace TNL::Benchmarks::MatrixAssembly
