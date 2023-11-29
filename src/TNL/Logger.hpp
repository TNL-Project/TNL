// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/Logger.h>
#include <TNL/Backend.h>
#include <TNL/SystemInfo.h>

namespace TNL {

inline void
Logger::writeHeader( const std::string& title )
{
   const int fill = stream.fill();
   const int titleLength = title.length();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "|" << std::setw( width / 2 + titleLength / 2 ) << title << std::setw( width / 2 - titleLength / 2 ) << "|"
          << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline void
Logger::writeSeparator()
{
   const int fill = stream.fill();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline bool
Logger::writeSystemInformation( bool printGPUInfo )
{
   writeParameter< std::string >( "Host name:", getHostname() );
   writeParameter< std::string >( "System:", getSystemName() );
   writeParameter< std::string >( "Release:", getSystemRelease() );
   writeParameter< std::string >( "Architecture:", getSystemArchitecture() );
   writeParameter< std::string >( "TNL compiler:", getCompilerName() );
   const int threads = getCPUInfo().threads;
   const int cores = getCPUInfo().cores;
   int threadsPerCore = 0;
   if( cores > 0 )
      threadsPerCore = threads / cores;
   writeParameter< std::string >( "CPU info", "" );
   writeParameter< std::string >( "Model name:", getCPUInfo().modelName, 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< double >( "Max clock rate (in MHz):", getCPUMaxFrequency() / 1000, 1 );
   const CPUCacheSizes cacheSizes = getCPUCacheSizes();
   const auto cacheInfo = std::to_string( cacheSizes.L1data ) + ", " + std::to_string( cacheSizes.L1instruction ) + ", "
                        + std::to_string( cacheSizes.L2 ) + ", " + std::to_string( cacheSizes.L3 );
   writeParameter< std::string >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );

   if( printGPUInfo ) {
      writeParameter< std::string >( "GPU info", "" );
      // TNL supports using more than one device for computations only via MPI.
      // Hence, we print only the active device here.
      const int i = Backend::getDevice();
      writeParameter< std::string >( "Name", Backend::getDeviceName( i ), 1 );
      const auto deviceArch =
         std::to_string( Backend::getArchitectureMajor( i ) ) + "." + std::to_string( Backend::getArchitectureMinor( i ) );
      writeParameter< std::string >( "Architecture", deviceArch, 1 );
      writeParameter< int >( "GPU cores", Backend::getDeviceCores( i ), 1 );
      const double clockRate = Backend::getClockRate( i ) / 1.0e3;
      writeParameter< double >( "Clock rate (in MHz)", clockRate, 1 );
      const double globalMemory = Backend::getGlobalMemorySize( i ) / 1.0e9;
      writeParameter< double >( "Global memory (in GB)", globalMemory, 1 );
      const double memoryClockRate = Backend::getMemoryClockRate( i ) / 1.0e3;
      writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 1 );
      writeParameter< bool >( "ECC enabled", Backend::getECCEnabled( i ), 1 );
   }
   return true;
}

inline void
Logger::writeCurrentTime( const char* label )
{
   writeParameter< std::string >( label, getCurrentTime() );
}

template< typename T >
void
Logger::writeParameter( const std::string& label,
                        const std::string& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel )
{
   writeParameter( label, parameters.getParameter< T >( parameterName ), parameterLevel );
}

template< typename T >
void
Logger::writeParameter( const std::string& label, const T& value, int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i++ )
      stream << " ";
   std::stringstream str;
   str << value;
   stream << label << std::setw( width - label.length() - parameterLevel - 3 ) << str.str() << " |" << std::endl;
}

}  // namespace TNL
