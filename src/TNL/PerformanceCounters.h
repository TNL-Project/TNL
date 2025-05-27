// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>

namespace TNL {

/**
 * \brief Performance counter for measuring CPU cycles.
 *
 * The counter can be paused by calling \ref stop and \ref
 * start methods and reset by calling \ref reset.
 *
 * \par Example
 * \include PerformanceCountersExample.cpp
 * \par Output
 * \include PerformanceCountersExample.out
 */
struct PerformanceCounters
{
   //! \brief Constructor with no parameters.
   PerformanceCounters();

   //! \brief Reset counters.
   void
   reset();

   /**
    * \brief Starts counters.
    *
    * This method can be used also after using the \ref stop
    * method. The counters then continue the measuring.
    */
   void
   start();

   //! \brief Stops (pauses) the counters but do not set them to zeros.
   void
   stop();

   /**
    * \brief Returns the number of CPU cycles (machine cycles) elapsed on this timer.
    *
    * CPU cycles are counted by adding the number of CPU cycles between
    * \ref start and \ref stop methods together.
    */
   [[nodiscard]] unsigned long long int
   getCPUCycles() const;

   /**
    * \brief Writes a record into the \e logger.
    *
    * \param logger Name of Logger object.
    * \param logLevel A non-negative integer recording the log record indent.
    *
    * \par Example
    * \include TimerExampleLogger.cpp
    * \par Output
    * \include TimerExampleLogger.out
    */
   bool
   writeLog( Logger& logger, int logLevel = 0 ) const;

private:
   bool stopState = false;

   unsigned long long int initialCPUCycles = 0, totalCPUCycles = 0;

   [[nodiscard]] static unsigned long long int
   readCPUCycles();
};

}  // namespace TNL

#include <TNL/PerformanceCounters.hpp>
