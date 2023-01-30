// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
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

#ifdef SPY_OS_IS_MACOS
   #include <sys/types.h>
   #include <sys/sysctl.h>
   #include <regex>
#endif

#ifdef SPY_OS_IS_WINDOWS
   #include <windows.h>
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
   int cacheLineSize = 0;
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

   file.close();
   return info;
#elif defined( SPY_OS_IS_MACOS )
   CPUInfo info;

   // It seems that MacOS does not provide number of physical processors just number of cores.
   // With Apple Silicon, all systems are single CPU based.
   info.numberOfProcessors = 1;

   // Get model name
   std::array< char, 1024 > buffer;
   size_t buffer_size = buffer.size() * sizeof( char );
   sysctlbyname( "machdep.cpu.brand_string", buffer.data(), &buffer_size, NULL, 0 );
   info.modelName = buffer.data();

   // Get number of cores
   size_t cores_size = sizeof( info.cores );
   sysctlbyname( "hw.physicalcpu", &info.cores, &cores_size, NULL, 0 );

   size_t threads_size = sizeof( info.threads );
   sysctlbyname( "hw.logicalcpu", &info.threads, &threads_size, NULL, 0 );

   return info;

#elif defined( SPY_OS_IS_WINDOWS )
   // Get number of cores
   // https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
   SYSTEM_INFO sysinfo;
   GetSystemInfo( &sysinfo );
   info.core = sysinfo.dwNumberOfProcessors;
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

#ifdef SPY_OS_IS_MACOS
inline int
getCacheSize( const char* cache_type )
{
   std::array< char, 128 > buffer;
   std::string result;
   std::string command( "sysctl hw." );
   command += cache_type;
   std::unique_ptr< FILE, decltype( &pclose ) > pipe( popen( command.data(), "r" ), pclose );

   if( ! pipe ) {
      std::string msg = "Cannot call command '" + command + "' to detect cache size.";
      throw std::runtime_error( msg.data() );
   }

   while( fgets( buffer.data(), buffer.size(), pipe.get() ) != nullptr ) {
      result += buffer.data();
   }
   // L3cachesize can return empty string if it is missing
   if( result == "" )
      return 0;
   std::string regex_str( "hw." );
   regex_str += cache_type;
   regex_str += ": (\\d+)";
   std::regex re( regex_str.data() );
   std::smatch match;
   if( ! std::regex_search( result, match, re ) && match.size() > 1 ) {
      std::string msg = "Cannot parse output of sysctl command: " + result;
      throw std::runtime_error( msg.data() );
   }
   int cacheSize = std::stoi( match[ 1 ].str() );
   return cacheSize;
}
#endif

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
   sizes.cacheLineSize = detail::readFile< int >( "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size" );
   return sizes;
#elif defined( SPY_OS_IS_MACOS )
   CPUCacheSizes sizes;
   sizes.L1instruction = getCacheSize( "l1icachesize" );
   sizes.L1data = getCacheSize( "l1dcachesize" );
   sizes.L2 = getCacheSize( "l2cachesize" );
   sizes.L3 = getCacheSize( "l3cachesize" );

   // Get cache lines size
   std::string result;
   std::unique_ptr< FILE, decltype( &pclose ) > pipe( popen( "sysctl hw.cachelinesize", "r" ), pclose );

   if( ! pipe )
      throw std::runtime_error( "Cannot call sysctl command to detect the cache line size." );

   std::array< char, 1024 > buffer;
   while( fgets( buffer.data(), buffer.size(), pipe.get() ) != nullptr ) {
      result += buffer.data();
   }
   std::regex re( "hw.cachelinesize: (\\d+)" );
   std::smatch match;
   if( std::regex_search( result, match, re ) && match.size() > 1 )
      sizes.cacheLineSize = std::stoi( match[ 1 ].str() );
   else
      throw std::runtime_error( "Failed to parse output of sysctl to detect the cache line size." );

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
