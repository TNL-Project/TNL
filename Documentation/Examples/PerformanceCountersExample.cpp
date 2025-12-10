#include <iostream>
#include <chrono>
#include <thread>

#include <TNL/PerformanceCounters.h>

int
main()
{
   const int milliseconds = 0.5e3;
   TNL::PerformanceCounters performanceCounters;
   performanceCounters.start();
   std::this_thread::sleep_for( std::chrono::milliseconds( milliseconds ) );
   performanceCounters.stop();

   std::cout << "Elapsed CPU cycles: " << performanceCounters.getCPUCycles() << '\n';
   performanceCounters.reset();
   std::cout << "CPU cycles after reset: " << performanceCounters.getCPUCycles() << '\n';
}
