// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>

namespace TNL {

class Logger;

/**
 * \brief Class for real time, CPU time and CPU cycles measuring.
 *
 * It measures the elapsed real time and CPU time (in seconds)
 * elapsed on the timer. The timer can be paused by calling \ref stop and \ref
 * start methods and reseted by calling \ref reset.
 *
 * \par Example
 * \include TimerExample.cpp
 * \par Output
 * \include TimerExample.out
 */
class Timer
{
public:
   //! \brief Basic constructor creating a new timer and resets it.
   Timer();

   //! \brief Reset the CPU and real-time timers.
   void
   reset();

   //! \brief Stops (pauses) the CPU and the real-time timers, but does not set them to zero.
   void
   stop();

   /**
    * \brief Starts timer.
    *
    * Starts the CPU and real-time timers. This method can be used also after using the \ref stop
    * method. The timer then continues measuring the time without reseting.
    */
   void
   start();

   /**
    * \brief Returns the elapsed real time on this timer.
    *
    * This method returns the real time elapsed so far (in seconds).
    * This method can be called while the timer is running, there is no
    * need to use \ref stop method first.
    */
   [[nodiscard]] double
   getRealTime() const;

   /**
    * \brief Returns the elapsed CPU time on this timer.
    *
    * This method returns the CPU time (i.e. time the CPU spent by processing
    * this process) elapsed so far (in seconds). This method can be called
    * while the timer is running, there is no need to use \ref stop method
    * first.
    */
   [[nodiscard]] double
   getCPUTime() const;

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

protected:
   using TimePoint = typename std::chrono::high_resolution_clock::time_point;
   using Duration = typename std::chrono::high_resolution_clock::duration;

   //! \brief Function for measuring the real time.
   static TimePoint
   readRealTime();

   //! \brief Function for measuring the CPU time.
   static double
   readCPUTime();

   //! \brief Converts the real time into seconds as a floating point number.
   static double
   durationToDouble( const Duration& duration );

   TimePoint initialRealTime;

   Duration totalRealTime;

   double initialCPUTime, totalCPUTime;

   bool stopState;
};

}  // namespace TNL

#include <TNL/Timer.hpp>
