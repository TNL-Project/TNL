#include <gtest/gtest.h>
#include "GtestPrintToOverrides.h"

// when a test is split into several translation units (see TypedTestSplit.h), only the first part defines main
#if ! defined( TNL_TEST_PART ) || TNL_TEST_PART == 0
int
main( int argc, char* argv[] )
{
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
}
#endif
