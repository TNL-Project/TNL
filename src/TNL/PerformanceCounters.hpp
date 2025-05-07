// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/PerformanceCounters.h>

#include <TNL/3rdparty/spy.hpp>

#ifdef SPY_OS_IS_MACOS
   #include <TNL/3rdparty/kperf.h>
#endif

namespace TNL {

inline PerformanceCounters::PerformanceCounters()
{
#ifdef SPY_OS_IS_MACOS
   if( kperf_init() != 0 )
      throw std::runtime_error( "Cannot initialize kperf for performance counters." );
#endif
   reset();
}

inline void
PerformanceCounters::reset()
{
   this->initialCPUCycles = readCPUCycles();
   this->totalCPUCycles = 0;
   this->stopState = true;
}

inline void
PerformanceCounters::stop()
{
   if( ! this->stopState ) {
      this->totalCPUCycles += readCPUCycles() - this->initialCPUCycles;
      this->stopState = true;
   }
}

inline void
PerformanceCounters::start()
{
   this->initialCPUCycles = readCPUCycles();
   this->stopState = false;
}

inline unsigned long long int
PerformanceCounters::getCPUCycles() const
{
   if( ! this->stopState )
      return readCPUCycles() - this->initialCPUCycles;
   return this->totalCPUCycles;
}

inline bool
PerformanceCounters::writeLog( Logger& logger, int logLevel ) const
{
   logger.writeParameter< unsigned long long int >( "CPU Cycles:", this->getCPUCycles(), logLevel );
   return true;
}

inline unsigned long long int
PerformanceCounters::readCPUCycles()
{
#if defined( SPY_OS_IS_LINUX )  // TODO: Does it work even on Windows?
   unsigned hi;
   unsigned lo;
   __asm__ __volatile__( "rdtsc" : "=a"( lo ), "=d"( hi ) );
   return ( (unsigned long long) lo ) | ( ( (unsigned long long) hi ) << 32 );
#elif defined( SPY_OS_IS_MACOS )
   unsigned long long int cpu_cycles;
   unsigned long long int instructions;
   unsigned long long int branches;
   unsigned long long int branch_misses;
   get_kperf_counters( cpu_cycles, instructions, branches, branch_misses );
   return cpu_cycles;
#else
   return 0;
#endif
}

}  // namespace TNL
