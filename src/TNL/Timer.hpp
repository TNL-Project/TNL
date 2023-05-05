// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/3rdparty/spy.hpp>

#if defined( SPY_OS_IS_LINUX ) || defined( SPY_OS_IS_MACOS )
   #include <sys/resource.h>
#endif

namespace TNL {

inline Timer::Timer()
{
   reset();
}

inline void
Timer::reset()
{
   this->initialCPUTime = 0;
   this->totalCPUTime = 0.0;
   this->initialRealTime = TimePoint();
   this->totalRealTime = Duration();
   this->initialCPUCycles = 0;
   this->totalCPUCycles = 0;
   this->stopState = true;
}

inline void
Timer::stop()
{
   if( ! this->stopState ) {
      this->totalRealTime += readRealTime() - this->initialRealTime;
      this->totalCPUTime += readCPUTime() - this->initialCPUTime;
      this->totalCPUCycles += readCPUCycles() - this->initialCPUCycles;
      this->stopState = true;
   }
}

inline void
Timer::start()
{
   this->initialRealTime = readRealTime();
   this->initialCPUTime = readCPUTime();
   this->initialCPUCycles = readCPUCycles();
   this->stopState = false;
}

inline double
Timer::getRealTime() const
{
   if( ! this->stopState )
      return durationToDouble( readRealTime() - this->initialRealTime );
   return durationToDouble( this->totalRealTime );
}

inline double
Timer::getCPUTime() const
{
   if( ! this->stopState )
      return readCPUTime() - this->initialCPUTime;
   return this->totalCPUTime;
}

inline unsigned long long int
Timer::getCPUCycles() const
{
   if( ! this->stopState )
      return readCPUCycles() - this->initialCPUCycles;
   return this->totalCPUCycles;
}

inline bool
Timer::writeLog( Logger& logger, int logLevel ) const
{
   logger.writeParameter< double >( "Real time:", this->getRealTime(), logLevel );
   logger.writeParameter< double >( "CPU time:", this->getCPUTime(), logLevel );
   logger.writeParameter< unsigned long long int >( "CPU Cycles:", this->getCPUCycles(), logLevel );
   return true;
}

inline typename Timer::TimePoint
Timer::readRealTime()
{
   return std::chrono::high_resolution_clock::now();
}

inline double
Timer::readCPUTime()
{
#if defined( SPY_OS_IS_LINUX ) || defined( SPY_OS_IS_MACOS )
   rusage initUsage;
   getrusage( RUSAGE_SELF, &initUsage );
   return initUsage.ru_utime.tv_sec + 1.0e-6 * (double) initUsage.ru_utime.tv_usec;
#else
   return -1;
#endif
}

inline unsigned long long int
Timer::readCPUCycles()
{
#ifdef SPY_OS_IS_LINUX
   unsigned hi;
   unsigned lo;
   __asm__ __volatile__( "rdtsc" : "=a"( lo ), "=d"( hi ) );
   return ( (unsigned long long) lo ) | ( ( (unsigned long long) hi ) << 32 );
#else
   // TODO: implement for Windows and macOS:
   // https://lemire.me/blog/2021/03/24/counting-cycles-and-instructions-on-the-apple-m1-processor/
   return 0;
#endif
}

inline double
Timer::durationToDouble( const Duration& duration )
{
   std::chrono::duration< double > dur( duration );
   return dur.count();
}

}  // namespace TNL
