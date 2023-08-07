#ifdef NDEBUG
   #undef NDEBUG
   #define HIP_ENABLE_PRINTF
#endif

#include <TNL/Assert.h>
#include <TNL/Exceptions/BackendRuntimeError.h>

#include <gtest/gtest.h>

using namespace TNL;

// clang-format off
#define WRAP_ASSERT( suffix, statement, not_failing )                                                        \
__global__                                                                                                   \
void kernel_##suffix( int* output )                                                                          \
{                                                                                                            \
   const bool tr = true;                                                                                     \
   const bool fa = false;                                                                                    \
   const int two = 2;                                                                                        \
   const int ten = 10;                                                                                       \
   /* pointers */                                                                                            \
   const double* data_null = nullptr;                                                                        \
   const double** data_full = &data_null;                                                                    \
                                                                                                             \
   /* ignore warnings due to potentially unused variables */                                                 \
   (void) tr;                                                                                                \
   (void) fa;                                                                                                \
   (void) two;                                                                                               \
   (void) ten;                                                                                               \
   (void) data_null;                                                                                         \
   (void) data_full;                                                                                         \
                                                                                                             \
   statement;                                                                                                \
                                                                                                             \
   /* actually do something to test if the execution control gets here */                                    \
   output[ 0 ] = 1;                                                                                          \
}                                                                                                            \
                                                                                                             \
TEST( AssertTestHip, suffix )                                                                                \
{                                                                                                            \
   int* output_hip;                                                                                          \
   int output_host = 0;                                                                                      \
                                                                                                             \
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &output_hip, sizeof( int ) ) );                                \
   TNL_BACKEND_SAFE_CALL( hipMemcpy( output_hip, &output_host, sizeof( int ), hipMemcpyHostToDevice ) );     \
                                                                                                             \
   kernel_##suffix<<< 1, 1 >>>( output_hip );                                                                \
   if( not_failing ) {                                                                                       \
      EXPECT_NO_THROW( TNL_BACKEND_SAFE_CALL( hipDeviceSynchronize() ) );                                    \
      TNL_BACKEND_SAFE_CALL( hipMemcpy( &output_host, output_hip, sizeof( int ), hipMemcpyDeviceToHost ) );  \
      EXPECT_EQ( output_host, 1 );                                                                           \
   }                                                                                                         \
   else                                                                                                      \
      EXPECT_THROW( TNL_BACKEND_SAFE_CALL( hipDeviceSynchronize() ), TNL::Exceptions::BackendRuntimeError ); \
                                                                                                             \
   TNL_BACKEND_SAFE_CALL( hipFree( output_hip ) );                                                           \
}
// clang-format on

// not failing statements:
WRAP_ASSERT( test1, TNL_ASSERT_TRUE( true, "true is true" ), true )
WRAP_ASSERT( test2, TNL_ASSERT_TRUE( tr, "true is true" ), true )
WRAP_ASSERT( test3, TNL_ASSERT_FALSE( false, "false is false" ), true )
WRAP_ASSERT( test4, TNL_ASSERT_FALSE( fa, "false is false" ), true )

WRAP_ASSERT( test5, TNL_ASSERT_EQ( two, 2, "two is 2" ), true )
WRAP_ASSERT( test6, TNL_ASSERT_NE( ten, 2, "ten is not 2" ), true )
WRAP_ASSERT( test7, TNL_ASSERT_LT( two, 10, "two < 10" ), true )
WRAP_ASSERT( test8, TNL_ASSERT_LE( two, 10, "two <= 10" ), true )
WRAP_ASSERT( test9, TNL_ASSERT_LE( two, 2, "two <= 2" ), true )
WRAP_ASSERT( test10, TNL_ASSERT_GT( ten, 2, "ten > 2" ), true )
WRAP_ASSERT( test11, TNL_ASSERT_GE( ten, 10, "ten >= 10" ), true )
WRAP_ASSERT( test12, TNL_ASSERT_GE( ten, 2, "ten >= 2" ), true )

WRAP_ASSERT( test13, TNL_ASSERT_FALSE( data_null, "nullptr is false" ), true )
WRAP_ASSERT( test14, TNL_ASSERT_TRUE( data_full, "non-nullptr is true" ), true )

// errors:
#ifndef __HIP__
// FIXME: we can't test errors on HIP, because abort() aborts the whole program, not just the kernel
WRAP_ASSERT( test15, TNL_ASSERT_TRUE( false, "false is true" );, false )
WRAP_ASSERT( test16, TNL_ASSERT_TRUE( fa, "false is true" );, false )
WRAP_ASSERT( test17, TNL_ASSERT_FALSE( true, "true is false" );, false )
WRAP_ASSERT( test18, TNL_ASSERT_FALSE( tr, "true is false" );, false )

WRAP_ASSERT( test19, TNL_ASSERT_NE( two, 2, "two != 2" );, false )
WRAP_ASSERT( test20, TNL_ASSERT_EQ( ten, 2, "ten == 2" );, false )
WRAP_ASSERT( test21, TNL_ASSERT_GE( two, 10, "two >= 10" );, false )
WRAP_ASSERT( test22, TNL_ASSERT_GT( two, 10, "two > 10" );, false )
WRAP_ASSERT( test23, TNL_ASSERT_GT( two, 2, "two > 2" );, false )
WRAP_ASSERT( test24, TNL_ASSERT_LE( ten, 2, "ten <= 2" );, false ;
WRAP_ASSERT( test25, TNL_ASSERT_LT( ten, 10, "ten < 10" );, false )
WRAP_ASSERT( test26, TNL_ASSERT_LT( ten, 2, "ten < 2" );, false )

WRAP_ASSERT( test27, TNL_ASSERT_TRUE( data_null, "nullptr is true" );, false )
WRAP_ASSERT( test28, TNL_ASSERT_FALSE( data_full, "non-nullptr is false" );, false )
#endif

#include "main.h"
