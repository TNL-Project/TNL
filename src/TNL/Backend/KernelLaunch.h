// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "Types.h"
#include "Functions.h"
#include <TNL/Exceptions/BackendRuntimeError.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/TypeInfo.h>

namespace TNL::Backend {

/**
 * \brief Holds the parameters necessary to \e launch a CUDA or HIP kernel
 * (i.e. schedule it for execution on some stream of some device).
 */
struct LaunchConfiguration
{
   //! \brief kernel grid dimensions (in blocks)
   dim3 gridSize;

   //! \brief kernel block dimensions (in threads)
   dim3 blockSize;

   //! \brief size of dynamic shared memory (in bytes per block)
   std::size_t dynamicSharedMemorySize = 0U;

   //! \brief stream handle
   Backend::stream_t stream = 0;  // NOLINT(modernize-use-nullptr)

   //! \brief indicates whether host execution is blocked until the kernel execution is finished
   bool blockHostUntilFinished = true;

   LaunchConfiguration() = default;
   constexpr LaunchConfiguration( const LaunchConfiguration& ) = default;
   constexpr LaunchConfiguration( LaunchConfiguration&& ) = default;

   constexpr LaunchConfiguration( dim3 gridSize,
                                  dim3 blockSize,
                                  std::size_t dynamicSharedMemorySize = 0U,
                                  Backend::stream_t stream = 0,  // NOLINT(modernize-use-nullptr)
                                  bool blockHostUntilFinished = true )
   : gridSize( gridSize ),
     blockSize( blockSize ),
     dynamicSharedMemorySize( dynamicSharedMemorySize ),
     stream( stream ),
     blockHostUntilFinished( blockHostUntilFinished )
   {}
};

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernel( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   static_assert( std::is_function_v< RawKernel >
                     || (std::is_pointer_v< RawKernel > && std::is_function_v< ::std::remove_pointer_t< RawKernel > >),
                  "Only a plain function or function pointer can be launched as a CUDA kernel. "
                  "You are attempting to launch something else." );

   if( kernel_function == nullptr )
      throw std::logic_error( "cannot call a function via nullptr" );

   // TODO: basic verification of the configuration

#ifdef TNL_DEBUG_KERNEL_LAUNCHES
   // clang-format off
   std::cout << "Type of kernel function: " << TNL::getType( kernel_function ) << "\n";
   std::cout << "Kernel launch configuration:\n"
             << "\t- grid size: " << launch_configuration.gridSize.x << " x "
                                  << launch_configuration.gridSize.y << " x "
                                  << launch_configuration.gridSize.z << "\n"
             << "\t- block size: " << launch_configuration.blockSize.x << " x "
                                   << launch_configuration.blockSize.y << " x "
                                   << launch_configuration.blockSize.z
             << "\n"
             << "\t- stream: " << launch_configuration.stream << "\n"
             << "\t- dynamic shared memory size: " << launch_configuration.dynamicSharedMemorySize << "\n";
   std::cout.flush();
   // clang-format on
#endif

#if defined( __CUDACC__ ) || defined( __HIP__ )
   // FIXME: clang-format 13.0.0 is still inserting spaces between "<<<" and ">>>":
   // https://github.com/llvm/llvm-project/issues/52881
   // clang-format off
   kernel_function <<<
         launch_configuration.gridSize,
         launch_configuration.blockSize,
         launch_configuration.dynamicSharedMemorySize,
         launch_configuration.stream
      >>>( ::std::forward< KernelParameters >( parameters )... );
   // clang-format on

   if( launch_configuration.blockHostUntilFinished )
      Backend::streamSynchronize( launch_configuration.stream );

   // use custom error handling instead of TNL_CHECK_CUDA_DEVICE
   // to add the kernel function type to the error message
   #if defined( __CUDACC__ )
   const Backend::error_t status = cudaGetLastError();
   if( status != cudaSuccess ) {
      std::string msg = "detected after launching kernel " + TNL::getType( kernel_function ) + "\nSource: line "
                      + std::to_string( __LINE__ ) + " in " + __FILE__;
      throw Exceptions::BackendRuntimeError( status, msg );
   }
   #elif defined( __HIP__ )
   const Backend::error_t status = hipGetLastError();
   if( status != hipSuccess ) {
      std::string msg = "detected after launching kernel " + TNL::getType( kernel_function ) + "\nSource: line "
                      + std::to_string( __LINE__ ) + " in " + __FILE__;
      throw Exceptions::BackendRuntimeError( status, msg );
   }
   #endif
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernelSync( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   launch_configuration.blockHostUntilFinished = true;
   launchKernel( kernel_function, launch_configuration, std::forward< KernelParameters >( parameters )... );
}

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernelAsync( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   launch_configuration.blockHostUntilFinished = false;
   launchKernel( kernel_function, launch_configuration, std::forward< KernelParameters >( parameters )... );
}

}  // namespace TNL::Backend
