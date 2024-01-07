// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <unordered_map>

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

[[nodiscard]] inline int
getClockRate( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.clockRate;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.clockRate;
#else
   throw Exceptions::BackendSupportMissing();
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

[[nodiscard]] inline int
getMemoryClockRate( int deviceNum )
{
#if defined( __CUDACC__ )
   cudaDeviceProp properties;
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceProperties( &properties, deviceNum ) );
   return properties.memoryClockRate;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.memoryClockRate;
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
   return properties.ECCEnabled;
#elif defined( __HIP__ )
   hipDeviceProp_t properties;
   TNL_BACKEND_SAFE_CALL( hipGetDeviceProperties( &properties, deviceNum ) );
   return properties.ECCEnabled;
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
