// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "Macros.h"
#include <TNL/Exceptions/BackendSupportMissing.h>

namespace TNL::Backend {

[[nodiscard]] inline std::string
getDeviceName( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.name;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.name;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline int
getArchitectureMajor( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.major;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.major;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline int
getArchitectureMinor( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.minor;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.minor;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

/**
 * \brief Checks whether programmatic dependent launch can be used for a kernel.
 *
 * CUDA requires both a device with compute capability at least 9.0 and a
 * kernel compiled for that architecture. The result is cached per device for
 * each kernel signature. HIP and non-GPU builds always return `false`.
 *
 * \tparam RawKernel is the kernel function pointer type.
 * \param kernel_function is the kernel to be launched.
 * \param deviceNum is the device on which the kernel will run.
 * \return `true` when the launch can use programmatic stream serialization.
 */
template< typename RawKernel >
[[nodiscard]] inline bool
supportsProgrammaticDependentLaunch( RawKernel kernel_function, int deviceNum )
{
#if defined( __CUDACC__ )
   // the map is shared by all instantiations with the same kernel type (it
   // has vague linkage across translation units), so it must be keyed by
   // the function pointer as well as the device, and concurrent first calls
   // from multiple host threads are serialized
   static std::mutex mutex;
   static std::map< std::pair< const void*, int >, bool > results;
   std::lock_guard< std::mutex > lock( mutex );
   const auto key = std::make_pair( reinterpret_cast< const void* >( kernel_function ), deviceNum );
   if( auto it = results.find( key ); it != results.end() )
      return it->second;
   bool supported = false;
   if( getArchitectureMajor( deviceNum ) >= 9 ) {
      cudaFuncAttributes attributes;
      TNL_BACKEND_SAFE_CALL( cudaFuncGetAttributes( &attributes, reinterpret_cast< const void* >( kernel_function ) ) );
      supported = attributes.ptxVersion >= 90;
   }
   results.emplace( key, supported );
   return supported;
#else
   (void) kernel_function;
   (void) deviceNum;
   return false;
#endif
}

[[nodiscard]] inline std::size_t
getGlobalMemorySize( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.totalGlobalMem;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.totalGlobalMem;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline std::size_t
getFreeGlobalMemory()
{
#if defined( __CUDACC__ )
   std::size_t free = 0;
   std::size_t total = 0;
   TNL_BACKEND_SAFE_CALL( cudaMemGetInfo( &free, &total ) );
   return free;
#elif defined( __HIP__ )
   std::size_t free = 0;
   std::size_t total = 0;
   TNL_BACKEND_SAFE_CALL( hipMemGetInfo( &free, &total ) );
   return free;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline bool
getECCEnabled( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return static_cast< bool >( properties.ECCEnabled );
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return static_cast< bool >( properties.ECCEnabled );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline int
getDeviceMultiprocessors( int deviceNum )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // results are cached because they are used for configuration of some kernels
   static std::unordered_map< int, int > results;
   if( results.count( deviceNum ) == 0 ) {
   #if defined( __CUDACC__ )
      cudaDeviceProp properties;
      TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   #elif defined( __HIP__ )
      hipDeviceProp_t properties;
      TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   #endif
      results.emplace( deviceNum, properties.multiProcessorCount );
      return properties.multiProcessorCount;
   }
   return results[ deviceNum ];
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline int
getDeviceCoresPerMultiprocessors( int deviceNum )
{
#if defined( __CUDACC__ )
   int major = getArchitectureMajor( deviceNum );
   int minor = getArchitectureMinor( deviceNum );
   switch( major ) {
      case 1:  // Tesla generation, G80, G8x, G9x classes
         return 8;
      case 2:  // Fermi generation
         switch( minor ) {
            case 0:  // GF100 class
               return 32;
            case 1:  // GF10x class
               return 48;
            default:
               return -1;
         }
      case 3:  // Kepler generation -- GK10x, GK11x classes
         return 192;
      case 5:  // Maxwell generation -- GM10x, GM20x classes
         return 128;
      case 6:  // Pascal generation
         switch( minor ) {
            case 0:  // GP100 class
               return 64;
            case 1:  // GP10x classes
            case 2:
               return 128;
            default:
               return -1;
         }
      case 7:  // Volta and Turing generations
         return 64;
      case 8:  // Ampere generation
         switch( minor ) {
            case 0:  // GA100 class
               return 64;
            case 6:
               return 128;
            default:
               return -1;
         }
      default:
         return -1;
   }
#elif defined( __HIP__ )
   // TODO: check if this is general enough
   // 64 taken from https://en.wikipedia.org/wiki/Graphics_Core_Next#Compute_units
   return 64;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline int
getDeviceCores( int deviceNum )
{
   return getDeviceMultiprocessors( deviceNum ) * getDeviceCoresPerMultiprocessors( deviceNum );
}

[[nodiscard]] inline int
getRegistersPerMultiprocessor( int deviceNum )
{
#if defined( __CUDACC__ )
   // results are cached because they are used for configuration of some kernels
   static std::unordered_map< int, int > results;
   if( results.count( deviceNum ) == 0 ) {
      cudaDeviceProp properties;
      TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
      results.emplace( deviceNum, properties.regsPerMultiprocessor );
      return properties.regsPerMultiprocessor;
   }
   return results[ deviceNum ];
#elif defined( __HIP__ )
   // TODO: regsPerMultiprocessor is not part of hipDeviceProp_t yet.
   throw std::runtime_error( "HIP cannot detect number of registers per multiprocessor." );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline std::size_t
getSharedMemoryPerBlock( int deviceNum )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // results are cached because they are used for configuration of some kernels
   static std::unordered_map< int, int > results;
   if( results.count( deviceNum ) == 0 ) {
   #if defined( __CUDACC__ )
      cudaDeviceProp properties;
      TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   #elif defined( __HIP__ )
      hipDeviceProp_t properties;
      TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   #endif
      results.emplace( deviceNum, properties.sharedMemPerBlock );
      return properties.sharedMemPerBlock;
   }
   return results[ deviceNum ];
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

}  // namespace TNL::Backend
