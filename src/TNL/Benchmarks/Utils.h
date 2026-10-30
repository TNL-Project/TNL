// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

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
 * both at least `loops` iterations complete and the measured time reaches
 * `minTime` seconds. At least one iteration is always measured.
 * Computes mean, median, and standard deviation of the measurements.
 *
 * Before the timed loop begins, **warmup iterations** (controlled by the
 * `warmupLoops` and `warmupMinTime` parameters) are performed (calling
 * `reset()` followed by `compute()`). The warmup continues until both
 * conditions are satisfied: at least `warmupLoops` iterations have been
 * executed AND at least `warmupMinTime` seconds have elapsed. This serves
 * to stabilize thermal and frequency states and amortize one-time costs
 * such as CUDA JIT compilation. For CUDA devices, device synchronization
 * is performed after each warmup iteration to ensure the kernel has
 * completed before timing starts.
 *
 * If `forceWarmupLoop` is `false` and `minTime` is positive, a warmup iteration
 * whose computation takes at least `minTime` seconds ends the warmup and is
 * used as the first timing sample. Such a computation is long enough that the
 * effects compensated by the warmup are negligible, so it is not repeated
 * only to be discarded.
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
 * \param loops Minimum number of iterations
 * \param minTime Minimum total runtime in seconds
 * \param warmupLoops Number of untimed warmup iterations (0 to disable, unless warmupMinTime > 0)
 * \param warmupMinTime Minimum total warmup runtime in seconds (0 to disable)
 * \param forceWarmupLoop Whether warmup iterations are always discarded,
 *        even if they take at least `minTime` seconds
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
   double minTime,
   std::size_t warmupLoops,
   double warmupMinTime,
   bool forceWarmupLoop,
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

   // the number of samples is not known in advance when minTime is given
   std::vector< double > results_time;
   std::vector< long long int > results_cpu_cycles;
   double total_time = 0.0;

   // records the last measured iteration as a timing sample
   auto addSample = [ & ]()
   {
      results_time.push_back( timer.getRealTime() );
      total_time += timer.getRealTime();
      if constexpr( countCycles )
         results_cpu_cycles.push_back( performanceCounters.getCPUCycles() );
   };

   // warm up: untimed iterations to stabilize thermal/frequency states and amortize JIT overhead
   Timer warmupTimer;
   warmupTimer.start();
   for( std::size_t i = 0; i < warmupLoops || warmupTimer.getRealTime() < warmupMinTime; i++ ) {
      measureIteration();
      if( ! forceWarmupLoop && minTime > 0 && timer.getRealTime() >= minTime ) {
         // the computation is so long that the warmup effects are negligible, use it as a sample
         addSample();
         break;
      }
   }
   warmupTimer.stop();

   while( results_time.empty() || results_time.size() < loops || total_time < minTime ) {
      // abuse the monitor's "time" for loops
      monitor.setTime( results_time.size() + 1 );
      measureIteration();
      addSample();
   }

   const std::size_t samples = results_time.size();

   // sort for median computation (also improves summation accuracy)
   std::sort( results_time.begin(), results_time.end() );
   const double median_time = ( samples % 2 == 1 ) ? results_time[ samples / 2 ]
                                                   : ( results_time[ samples / 2 - 1 ] + results_time[ samples / 2 ] ) / 2.0;
   const double mean_time = std::accumulate( results_time.begin(), results_time.end(), 0.0 ) / static_cast< double >( samples );

   double median_cpu_cycles = 0.0;
   double mean_cpu_cycles = 0.0;
   if constexpr( countCycles ) {
      std::sort( results_cpu_cycles.begin(), results_cpu_cycles.end() );
      median_cpu_cycles =
         ( samples % 2 == 1 )
            ? static_cast< double >( results_cpu_cycles[ samples / 2 ] )
            : static_cast< double >( results_cpu_cycles[ samples / 2 - 1 ] + results_cpu_cycles[ samples / 2 ] ) / 2.0;
      mean_cpu_cycles =
         std::accumulate( results_cpu_cycles.begin(), results_cpu_cycles.end(), 0.0 ) / static_cast< double >( samples );
   }

   double stddev_time;
   double stddev_cpu_cycles;
   if( samples > 1 ) {
      double sum_time = 0.0;
      double sum_cpu_cycles = 0.0;
      for( std::size_t i = 0; i < samples; i++ ) {
         sum_time += ( results_time[ i ] - mean_time ) * ( results_time[ i ] - mean_time );
         if constexpr( countCycles )
            sum_cpu_cycles += ( results_cpu_cycles[ i ] - mean_cpu_cycles ) * ( results_cpu_cycles[ i ] - mean_cpu_cycles );
      }
      stddev_time = std::sqrt( sum_time / static_cast< double >( samples - 1 ) );
      stddev_cpu_cycles = std::sqrt( sum_cpu_cycles / static_cast< double >( samples - 1 ) );
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
