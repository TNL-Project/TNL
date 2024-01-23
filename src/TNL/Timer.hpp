// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
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
   this->stopState = true;
}

inline void
Timer::stop()
{
   if( ! this->stopState ) {
      this->totalRealTime += readRealTime() - this->initialRealTime;
      this->totalCPUTime += readCPUTime() - this->initialCPUTime;
      this->stopState = true;
   }
}

inline void
Timer::start()
{
   this->initialRealTime = readRealTime();
   this->initialCPUTime = readCPUTime();
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

inline bool
Timer::writeLog( Logger& logger, int logLevel ) const
{
   logger.writeParameter< double >( "Real time:", this->getRealTime(), logLevel );
   logger.writeParameter< double >( "CPU time:", this->getCPUTime(), logLevel );
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

inline double
Timer::durationToDouble( const Duration& duration )
{
   std::chrono::duration< double > dur( duration );
   return dur.count();
}

}  // namespace TNL
