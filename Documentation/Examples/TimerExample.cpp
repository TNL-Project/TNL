#include <iostream>
#include <chrono>
#include <thread>

#include <TNL/Timer.h>

int
main()
{
   const int milliseconds = 0.5e3;
   TNL::Timer time;
   time.start();
   std::this_thread::sleep_for( std::chrono::milliseconds( milliseconds ) );
   time.stop();

   std::cout << "Elapsed real time: " << time.getRealTime() << '\n';
   std::cout << "Elapsed CPU time: " << time.getCPUTime() << '\n';
   time.reset();
   std::cout << "Real time after reset:" << time.getRealTime() << '\n';
   std::cout << "CPU time after reset: " << time.getCPUTime() << '\n';
}
