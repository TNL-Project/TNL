// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/Logger.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/SystemInfo.h>

namespace TNL {

inline void
Logger::writeHeader( const String& title )
{
   const int fill = stream.fill();
   const int titleLength = title.getLength();
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
   writeParameter< String >( "Host name:", getHostname() );
   writeParameter< String >( "System:", getSystemName() );
   writeParameter< String >( "Release:", getSystemRelease() );
   writeParameter< String >( "Architecture:", getSystemArchitecture() );
   writeParameter< String >( "TNL compiler:", getCompilerName() );
   const int threads = getCPUInfo().threads;
   const int cores = getCPUInfo().cores;
   int threadsPerCore = 0;
   if( cores > 0 )
      threadsPerCore = threads / cores;
   writeParameter< String >( "CPU info", "" );
   writeParameter< String >( "Model name:", getCPUInfo().modelName, 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< double >( "Max clock rate (in MHz):", getCPUMaxFrequency() / 1000, 1 );
   const CPUCacheSizes cacheSizes = getCPUCacheSizes();
   const String cacheInfo = convertToString( cacheSizes.L1data ) + ", " + convertToString( cacheSizes.L1instruction ) + ", "
                          + convertToString( cacheSizes.L2 ) + ", " + convertToString( cacheSizes.L3 );
   writeParameter< String >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );

   if( printGPUInfo ) {
      writeParameter< String >( "CUDA GPU info", "" );
      // TODO: Printing all devices does not make sense until TNL can actually
      //       use more than one device for computations. Printing only the active
      //       device for now...
      //   int devices = getNumberOfDevices();
      //   writeParameter< int >( "Number of devices", devices, 1 );
      //   for( int i = 0; i < devices; i++ )
      //   {
      //      logger.writeParameter< int >( "Device no.", i, 1 );
      const int i = Cuda::DeviceInfo::getActiveDevice();
      writeParameter< String >( "Name", Cuda::DeviceInfo::getDeviceName( i ), 2 );
      const String deviceArch = convertToString( Cuda::DeviceInfo::getArchitectureMajor( i ) ) + "."
                              + convertToString( Cuda::DeviceInfo::getArchitectureMinor( i ) );
      writeParameter< String >( "Architecture", deviceArch, 2 );
      writeParameter< int >( "CUDA cores", Cuda::DeviceInfo::getCudaCores( i ), 2 );
      const double clockRate = (double) Cuda::DeviceInfo::getClockRate( i ) / 1.0e3;
      writeParameter< double >( "Clock rate (in MHz)", clockRate, 2 );
      const double globalMemory = (double) Cuda::DeviceInfo::getGlobalMemory( i ) / 1.0e9;
      writeParameter< double >( "Global memory (in GB)", globalMemory, 2 );
      const double memoryClockRate = (double) Cuda::DeviceInfo::getMemoryClockRate( i ) / 1.0e3;
      writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 2 );
      writeParameter< bool >( "ECC enabled", Cuda::DeviceInfo::getECCEnabled( i ), 2 );
      //   }
   }
   return true;
}

inline void
Logger::writeCurrentTime( const char* label )
{
   writeParameter< String >( label, getCurrentTime() );
}

template< typename T >
void
Logger::writeParameter( const String& label,
                        const String& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel )
{
   writeParameter( label, parameters.getParameter< T >( parameterName ), parameterLevel );
}

template< typename T >
void
Logger::writeParameter( const String& label, const T& value, int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i++ )
      stream << " ";
   std::stringstream str;
   str << value;
   stream << label << std::setw( width - label.getLength() - parameterLevel - 3 ) << str.str() << " |" << std::endl;
}

}  // namespace TNL
