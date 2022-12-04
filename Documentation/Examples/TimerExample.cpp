#include <iostream>
#include <chrono>
#include <thread>

#include <TNL/Timer.h>

int main()
{
   const int milliseconds = 0.5e3;
   TNL::Timer time;
   time.start();
   std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
   time.stop();

   std::cout << "Elapsed real time: " << time.getRealTime() << std::endl;
   std::cout << "Elapsed CPU time: " << time.getCPUTime() << std::endl;
   std::cout << "Elapsed CPU cycles: " << time.getCPUCycles() << std::endl;
   time.reset();
   std::cout << "Real time after reset:" << time.getRealTime() << std::endl;
   std::cout << "CPU time after reset: " << time.getCPUTime() << std::endl;
   std::cout << "CPU cycles after reset: " << time.getCPUCycles() << std::endl;
}
