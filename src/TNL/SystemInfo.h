// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <cstdlib>  // std::atoi
#include <ctime>    // std::localtime
#include <iomanip>  // std::put_time
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <TNL/3rdparty/spy.hpp>

#if defined( SPY_OS_IS_LINUX ) || defined( SPY_OS_IS_MACOS )
   #include <unistd.h>
   #include <sys/utsname.h>
   #include <sys/stat.h>
#endif

namespace TNL {

struct CPUInfo
{
   int numberOfProcessors = 0;
   std::string modelName;
   int threads = 0;
   int cores = 0;
};

struct CPUCacheSizes
{
   int L1instruction = 0;
   int L1data = 0;
   int L2 = 0;
   int L3 = 0;
};

namespace detail {

inline CPUInfo
parseCPUInfo()
{
#if defined( SPY_OS_IS_LINUX )
   CPUInfo info;
   std::ifstream file( "/proc/cpuinfo" );
   if( ! file ) {
      std::cerr << "Unable to read information from /proc/cpuinfo." << std::endl;
      return info;
   }

   std::set< int > processors;
   while( ! file.eof() ) {
      std::string line;
      std::getline( file, line );
      // check if line starts with "physical id"
      if( line.rfind( "physical id", 0 ) == 0 ) {
         std::size_t i = 0;
         while( i < line.size() && line[ i ] != ':' )
            i++;
         processors.insert( std::atoi( &line[ i + 1 ] ) );
         continue;
      }
      // FIXME: the rest does not work on heterogeneous multi-socket systems
      // check if line starts with "model name"
      if( line.rfind( "model name", 0 ) == 0 ) {
         std::size_t i = 0;
         while( i < line.size() && line[ i ] != ':' )
            i++;
         info.modelName = &line[ i + 1 ];
         continue;
      }
      // check if line starts with "cpu cores"
      if( line.rfind( "cpu cores", 0 ) == 0 ) {
         std::size_t i = 0;
         while( i < line.size() && line[ i ] != ':' )
            i++;
         info.cores = std::atoi( &line[ i + 1 ] );
         continue;
      }
      // check if line starts with "siblings"
      if( line.rfind( "siblings", 0 ) == 0 ) {
         std::size_t i = 0;
         while( i < line.size() && line[ i ] != ':' )
            i++;
         info.threads = atoi( &line[ i + 1 ] );
      }
   }
   info.numberOfProcessors = processors.size();

   return info;
#else
   return {};
#endif
}

template< typename ResultType >
ResultType
readFile( const std::string& fileName )
{
   std::ifstream file( fileName );
   if( ! file ) {
      std::cerr << "Unable to read information from " << fileName << "." << std::endl;
      return 0;  // NOLINT(modernize-use-nullptr)
   }
   ResultType result;
   file >> result;
   return result;
}

}  // namespace detail

inline std::string
getHostname()
{
#ifdef SPY_OS_IS_LINUX
   char host_name[ 256 ];
   gethostname( host_name, 255 );
   return host_name;
#else
   return "[unknown hostname]";
#endif
}

inline std::string
getSystemArchitecture()
{
   std::stringstream ss;
   ss << spy::architecture;
   return ss.str();
}

inline std::string
getSystemName()
{
   std::stringstream ss;
   ss << spy::operating_system;
   return ss.str();
}

inline std::string
getSystemRelease()
{
#ifdef SPY_OS_IS_LINUX
   utsname uts;
   uname( &uts );
   return uts.release;
#else
   return "[unknown release]";
#endif
}

inline std::string
getCompilerName()
{
#if defined( __NVCC__ )
   // TODO: this can be removed when SPY supports nvcc: https://github.com/jfalcou/spy/issues/31
   #define TNL_STRINGIFY_IMPL( x ) #x
   // indirection is necessary in order to expand macros in the argument
   #define TNL_STRINGIFY( x ) TNL_STRINGIFY_IMPL( x )
   return "Nvidia NVCC (" TNL_STRINGIFY( __CUDACC_VER_MAJOR__ ) "." TNL_STRINGIFY( __CUDACC_VER_MINOR__ ) "." TNL_STRINGIFY(
      __CUDACC_VER_BUILD__ ) ")";
   #undef TNL_STRINGIFY
   #undef TNL_STRINGIFY_IMPL
#else
   std::stringstream ss;
   ss << spy::compiler;
   return ss.str();
#endif
}

inline std::string
getCurrentTime( const char* format = "%a %b %d %Y, %H:%M:%S" )
{
   const std::time_t time_since_epoch = std::time( nullptr );
   std::tm* localtime = std::localtime( &time_since_epoch );
   std::stringstream ss;
   ss << std::put_time( localtime, format );
   return ss.str();
}

inline CPUInfo
getCPUInfo()
{
   static CPUInfo info;
   if( info.numberOfProcessors == 0 )
      info = detail::parseCPUInfo();
   return info;
}

inline std::string
getOnlineCPUs()
{
   if constexpr( spy::operating_system == spy::linux_ ) {
      return detail::readFile< std::string >( "/sys/devices/system/cpu/online" );
   }
   else {
      return "";
   }
}

inline int
getCPUMaxFrequency( int cpu_id = 0 )
{
   if constexpr( spy::operating_system == spy::linux_ ) {
      std::string fileName( "/sys/devices/system/cpu/cpu" );
      fileName += std::to_string( cpu_id ) + "/cpufreq/cpuinfo_max_freq";
      return detail::readFile< int >( fileName );
   }
   else {
      return 0;
   }
}

inline CPUCacheSizes
getCPUCacheSizes( int cpu_id = 0 )
{
#ifdef SPY_OS_IS_LINUX
   std::string directory( "/sys/devices/system/cpu/cpu" );
   directory += std::to_string( cpu_id ) + "/cache";

   CPUCacheSizes sizes;
   for( int i = 0; i <= 3; i++ ) {
      const std::string cache = directory + "/index" + std::to_string( i );

      // check if the directory exists
      struct stat st;
      if( stat( cache.c_str(), &st ) != 0 || ! S_ISDIR( st.st_mode ) )
         break;

      const int level = detail::readFile< int >( cache + "/level" );
      const auto type = detail::readFile< std::string >( cache + "/type" );
      const int size = detail::readFile< int >( cache + "/size" );

      if( level == 1 && type == "Instruction" )
         sizes.L1instruction = size;
      else if( level == 1 && type == "Data" )
         sizes.L1data = size;
      else if( level == 2 )
         sizes.L2 = size;
      else if( level == 3 )
         sizes.L3 = size;
   }
   return sizes;
#else
   return {};
#endif
}

inline std::size_t
getFreeMemory()
{
#if defined( SPY_OS_IS_LINUX ) || defined( SPY_OS_IS_MACOS )
   std::size_t pages = sysconf( _SC_PHYS_PAGES );
   std::size_t page_size = sysconf( _SC_PAGE_SIZE );
   return pages * page_size;
#else
   return -1;
#endif
}

}  // namespace TNL
