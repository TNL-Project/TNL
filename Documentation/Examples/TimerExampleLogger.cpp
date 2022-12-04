#include <iostream>
#include <chrono>
#include <thread>

#include <TNL/Timer.h>
#include <TNL/Logger.h>

int main()
{
   const int milliseconds = 0.5e3;
   TNL::Timer time;
   time.start();
   std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
   time.stop();

   TNL::Logger logger( 50, std::cout );
   time.writeLog( logger, 0 );
}
