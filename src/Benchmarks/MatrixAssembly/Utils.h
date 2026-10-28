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

/**
 * \brief Measures the assembly of a matrix on the host followed by its transfer to the GPU.
 *
 * Measured operations:
 *
 * - `host assembly`: assembling the matrix on the host,
 * - `transfer`: copying the matrix assembled on the host to the GPU,
 * - `host assembly + transfer`: both together.
 *
 * \param resetAll destroys both the host and the GPU matrix.
 * \param resetDevice destroys only the GPU matrix (the host matrix is kept for the transfer).
 * \param hostAssemble assembles the matrix on the host.
 * \param transfer copies the matrix assembled on the host to the GPU.
 */
template< typename Device, typename ResetAll, typename ResetDevice, typename HostAssemble, typename Transfer >
void
timeHostAssemblyAndTransfer(
   Benchmark& benchmark,
   const std::string& performer,
   ResetAll&& resetAll,
   ResetDevice&& resetDevice,
   HostAssemble& hostAssemble,
   Transfer& transfer )
{
   auto total = [ & ]()
   {
      hostAssemble();
      transfer();
   };
   timeOperation< Device >( benchmark, "host assembly", performer, resetAll, hostAssemble );
   // the host matrix is assembled after the last loop
   timeOperation< Device >( benchmark, "transfer", performer, resetDevice, transfer );
   timeOperation< Device >( benchmark, "host assembly + transfer", performer, resetAll, total );
}

//! \brief Reports the result of the verification of the assembled matrix.
inline void
reportVerification( Benchmark& benchmark, const std::string& performer, const std::string& error )
{
   if( ! error.empty() )
      benchmark.addErrorMessage( performer + ": verification of the assembled matrix failed: " + error );
}

}  // namespace TNL::Benchmarks::MatrixAssembly
