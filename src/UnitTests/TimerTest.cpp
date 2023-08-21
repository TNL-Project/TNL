#include <gtest/gtest.h>

#include <TNL/Timer.h>

using namespace TNL;

TEST( TimerTest, Constructor )
{
   Timer time;
   time.reset();
   EXPECT_EQ( time.getRealTime(), 0 );
   /*time.start();
   EXPECT_FALSE(time.stopState);

   time.stop();
   EXPECT_TRUE(time.stopState);

   EXPECT_NE(time.getRealTime(),0);*/
}

#include "main.h"
