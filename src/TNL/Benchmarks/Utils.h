// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#include <TNL/Backend.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/MPI/Comm.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/PerformanceCounters.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/SystemInfo.h>
#include <TNL/Timer.h>

namespace TNL::Benchmarks {

/**
 * \brief Times a compute function over multiple iterations.
 *
 * Executes the compute function repeatedly, collecting timing samples until
 * either `loops` iterations complete or the measured time reaches `maxTime`
 * seconds. At least one iteration is always measured.
 * Computes mean, median, and standard deviation of the measurements.
 *
 * Before the timed loop begins, **warmup iterations** (controlled by the
 * `warmupLoops` and `warmupMaxTime` parameters) are performed (calling
 * `reset()` followed by `compute()`). The warmup ends when either
 * `warmupLoops` iterations have been executed or `warmupMaxTime` seconds
 * have elapsed. This serves to stabilize thermal and frequency states and
 * amortize one-time costs such as CUDA JIT compilation. For CUDA devices,
 * device synchronization is performed after each warmup iteration to ensure
 * the kernel has completed before timing starts.
 *
 * The time limits `maxTime` and `warmupMaxTime` do not interrupt a running
 * computation. They are checked after each iteration and only prevent starting
 * the next one, so the total time of a phase may exceed its limit by up to one
 * iteration.
 *
 * If `reuseLongWarmup` is `true` and `maxTime` is positive, a warmup iteration
 * whose computation takes at least `maxTime` seconds ends the warmup and is
 * used as the only timing sample. Such a computation exhausts the time budget
 * by itself and it is long enough that the effects compensated by the warmup
 * are negligible, so it is not executed once more only to be discarded.
 *
 * For CUDA devices, explicit synchronization is performed before and after
 * each timed computation to ensure accurate timing.
 *
 * CPU cycle counting is only available for host devices (Sequential, Host).
 *
 * \tparam Device Device type (e.g., \ref TNL::Devices::Host, \ref TNL::Devices::Cuda)
 * \tparam ComputeFunction Callable containing code to benchmark
 * \tparam ResetFunction Callable to reset state before each iteration
 * \tparam Monitor Solver monitor for tracking progress
 * \tparam ResultType Type with `setTimeResults` method
 *
 * \param compute Function to benchmark
 * \param reset Function called before each compute iteration
 * \param loops Maximum number of iterations
 * \param maxTime Total runtime in seconds after which no further iteration is started (0 for no limit)
 * \param warmupLoops Maximum number of warmup iterations (0 to disable)
 * \param warmupMaxTime Total warmup runtime in seconds after which no further warmup iteration is started
 *        (0 for no limit)
 * \param reuseLongWarmup Whether a warmup iteration taking at least `maxTime` seconds
 *        is used as the timing sample
 * \param monitor Solver monitor instance (must not be null)
 * \param result Output object to receive timing data via `setTimeResults()`
 *
 * \par Statistics computed:
 * - **Mean**: arithmetic average of all samples
 * - **Median**: middle value of sorted samples (average of two middle values for even sample counts)
 * - **Standard deviation**: sample standard deviation using Bessel's correction
 */
template<
   typename Device,
   typename ComputeFunction,
   typename ResetFunction,
   typename Monitor = TNL::Solvers::IterativeSolverMonitor< double >,
   typename ResultType >
void
timeFunction(
   ComputeFunction compute,
   ResetFunction reset,
   std::size_t loops,
   double maxTime,
   std::size_t warmupLoops,
   double warmupMaxTime,
   bool reuseLongWarmup,
   Monitor& monitor,
   ResultType& result )
{
   constexpr bool countCycles = std::is_same_v< Device, Devices::Sequential > || std::is_same_v< Device, Devices::Host >;

   // the timer is constructed zero-initialized and stopped
   Timer timer;

   // set timer to the monitor
   monitor.setTimer( timer );

   PerformanceCounters performanceCounters;

   // executes one iteration, only the computation is measured by the timer and the performance counters
   auto measureIteration = [ & ]()
   {
      reset();

      // Explicit synchronization of the GPU device
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         Backend::deviceSynchronize();

      // reset timer and performance counters before each computation
      timer.reset();
      performanceCounters.reset();
      timer.start();
      performanceCounters.start();
      compute();
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         Backend::deviceSynchronize();
      timer.stop();
      performanceCounters.stop();
   };

   // at least one iteration is always measured
   const std::size_t maxSamples = loops > 0 ? loops : 1;
   Containers::Vector< double > results_time( maxSamples, 0.0 );
   Containers::Vector< long long int > results_cpu_cycles( maxSamples, 0 );
   std::size_t samples = 0;
   double results_total_time = 0.0;

   // records the last measured iteration as a timing sample
   auto addSample = [ & ]()
   {
      results_time[ samples ] = timer.getRealTime();
      if constexpr( countCycles )
         results_cpu_cycles[ samples ] = performanceCounters.getCPUCycles();
      results_total_time += timer.getRealTime();
      samples++;
   };

   // warm up: iterations to stabilize thermal/frequency states and amortize JIT overhead,
   // their times are discarded unless a long iteration is reused as the sample
   Timer warmupTimer;
   warmupTimer.start();
   for( std::size_t i = 0; i < warmupLoops && ( warmupMaxTime <= 0 || warmupTimer.getRealTime() < warmupMaxTime ); i++ ) {
      measureIteration();
      if( reuseLongWarmup && maxTime > 0 && timer.getRealTime() >= maxTime ) {
         // the computation exhausts the time budget by itself and the warmup effects are negligible, use it as the sample
         addSample();
         break;
      }
   }
   warmupTimer.stop();

   while( samples < maxSamples && ( maxTime <= 0 || results_total_time < maxTime ) ) {
      // abuse the monitor's "time" for loops
      monitor.setTime( samples + 1 );
      measureIteration();
      addSample();
   }

   // only the measured samples are used for the statistics
   auto times = results_time.getView( 0, samples );
   auto cpu_cycles = results_cpu_cycles.getView( 0, samples );

   // sort for median computation (also improves summation accuracy)
   double* time_begin = times.getData();
   double* time_end = time_begin + samples;
   std::sort( time_begin, time_end );
   const double median_time =
      ( samples % 2 == 1 ) ? time_begin[ samples / 2 ] : ( time_begin[ samples / 2 - 1 ] + time_begin[ samples / 2 ] ) / 2.0;

   double median_cpu_cycles = 0.0;
   if constexpr( countCycles ) {
      long long int* cycles_begin = cpu_cycles.getData();
      long long int* cycles_end = cycles_begin + samples;
      std::sort( cycles_begin, cycles_end );
      median_cpu_cycles = ( samples % 2 == 1 )
                           ? static_cast< double >( cycles_begin[ samples / 2 ] )
                           : static_cast< double >( cycles_begin[ samples / 2 - 1 ] + cycles_begin[ samples / 2 ] ) / 2.0;
   }

   const double mean_time = sum( times ) / static_cast< double >( samples );
   const double mean_cpu_cycles = sum( cpu_cycles ) / static_cast< double >( samples );

   double stddev_time;
   double stddev_cpu_cycles;
   if( samples > 1 ) {
      stddev_time = 1.0 / std::sqrt( samples - 1 ) * l2Norm( times - mean_time );
      stddev_cpu_cycles = 1.0 / std::sqrt( samples - 1 ) * l2Norm( cpu_cycles - mean_cpu_cycles );
   }
   else {
      stddev_time = std::numeric_limits< double >::quiet_NaN();
      stddev_cpu_cycles = std::numeric_limits< double >::quiet_NaN();
   }

   // SolverMonitor takes a pointer to the timer, but the timer is local
   // so we must unset it to avoid returning a dangling reference to the caller
   monitor.unsetTimer();

   result.setTimeResults( samples, mean_time, median_time, stddev_time, mean_cpu_cycles, median_cpu_cycles, stddev_cpu_cycles );
}

inline std::map< std::string, std::string >
getHardwareMetadata()
{
   const CPUCacheSizes cacheSizes = getCPUCacheSizes();
   const std::string cacheInfo = std::to_string( cacheSizes.L1data ) + ", " + std::to_string( cacheSizes.L1instruction ) + ", "
                               + std::to_string( cacheSizes.L2 ) + ", " + std::to_string( cacheSizes.L3 );
   std::string cpuMaxFrequency;
   try {
      cpuMaxFrequency = std::to_string( getCPUMaxFrequency() / 1e3 );
   }
   catch( const std::exception& e ) {
      cpuMaxFrequency = "N/A";
      std::cerr << "Warning: failed to determine the CPU max frequency: " << e.what() << '\n';
   }
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const int activeGPU = Backend::getDevice();
   const std::string deviceArch = std::to_string( Backend::getArchitectureMajor( activeGPU ) ) + "."
                                + std::to_string( Backend::getArchitectureMinor( activeGPU ) );
#endif

#ifdef HAVE_MPI
   int nproc = 1;
   // check if MPI was initialized (some benchmarks do not initialize MPI even when
   // they are built with HAVE_MPI and thus MPI::GetSize() cannot be used blindly)
   if( TNL::MPI::Initialized() )
      nproc = TNL::MPI::GetSize();
#endif

   std::map< std::string, std::string > metadata{
      { "host name", getHostname() },
      { "architecture", getSystemArchitecture() },
      { "system", getSystemName() },
      { "system release", getSystemRelease() },
      { "compiler", getCompilerName() },
      { "start time", getCurrentTime() },
#ifdef HAVE_MPI
      { "number of MPI processes", std::to_string( nproc ) },
#endif
      { "OpenMP enabled", Devices::Host::isOMPEnabled() ? "yes" : "no" },
      { "OpenMP threads", std::to_string( Devices::Host::getMaxThreadsCount() ) },
      { "CPU model name", getCPUInfo().modelName },
      { "CPU cores", std::to_string( getCPUInfo().cores ) },
      { "CPU threads per core", std::to_string( getCPUInfo().threads / getCPUInfo().cores ) },
      { "CPU max frequency (MHz)", cpuMaxFrequency },
      { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#if defined( __CUDACC__ ) || defined( __HIP__ )
      { "GPU name", Backend::getDeviceName( activeGPU ) },
      { "GPU architecture", deviceArch },
      { "GPU CUDA cores", std::to_string( Backend::getDeviceCores( activeGPU ) ) },
      { "GPU global memory (GB)", std::to_string( static_cast< double >( Backend::getGlobalMemorySize( activeGPU ) ) / 1e9 ) },
      { "GPU memory ECC enabled", TNL::convertToString( Backend::getECCEnabled( activeGPU ) ) },
#endif
   };

   return metadata;
}

inline void
writeMapAsJson( const std::map< std::string, std::string >& data, std::ostream& out )
{
   out << "{\n";
   for( auto it = data.begin(); it != data.end(); ) {
      out << "\t\"" << it->first << "\": \"" << it->second << "\"";
      // increment the iterator now to peek at the next element
      it++;
      // write a comma if there are still elements remaining
      if( it != data.end() )
         out << ",";
      out << "\n";
   }
   out << "}\n" << std::flush;
}

inline void
writeMapAsJson( const std::map< std::string, std::string >& data, std::string filename, const std::string& newExtension = "" )
{
   namespace fs = std::filesystem;

   if( ! newExtension.empty() ) {
      const fs::path oldPath = filename;
      const fs::path newPath = oldPath.parent_path() / ( oldPath.stem().string() + newExtension );
      filename = newPath.string();
   }

   std::ofstream file( filename );
   // enable exceptions
   file.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
   writeMapAsJson( data, file );
}

}  // namespace TNL::Benchmarks
