// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>

#ifdef NDEBUG

   // empty macros for optimized build

   /**
    * \brief Asserts that the vector expression \e val1 is elementwise less than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is elementwise less than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_ALL_LE( val1, val2, msg )
   /**
    * \brief Asserts that the vector expression \e val1 is elementwise less than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is elementwise less than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_ALL_LT( val1, val2, msg )
   /**
    * \brief Asserts that the vector expression \e val1 is elementwise greater than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is elementwise greater than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_ALL_GE( val1, val2, msg )
   /**
    * \brief Asserts that the vector expression \e val1 is elementwise greater than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is elementwise greater than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_ALL_GT( val1, val2, msg )

#else /* #ifdef NDEBUG */

namespace TNL::Assert {

   // A macro for implementing the helper functions needed to implement
   // TNL_ASSERT_ALL_??. It is here just to avoid copy-and-paste of similar code.
   #define TNL_IMPL_ALL_CMP_HELPER_( op_name, op_func, op )                                                               \
      template< typename T1, typename T2 >                                                                                \
      __cuda_callable__                                                                                                   \
      void cmpHelper##op_name( const char* assertion,                                                                     \
                               const char* message,                                                                       \
                               const char* file,                                                                          \
                               const char* function,                                                                      \
                               int line,                                                                                  \
                               const char* expr1,                                                                         \
                               const char* expr2,                                                                         \
                               const T1& val1,                                                                            \
                               const T2& val2 )                                                                           \
      {                                                                                                                   \
         if( ! all( op_func( ( val1 ), ( val2 ) ) ) )                                                                     \
            ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, expr2, val1, val2, #op ); \
      }

// Implements the helper function for TNL_ASSERT_ALL_LE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_ALL_CMP_HELPER_( ALL_LE, lessEqual, <= )
// Implements the helper function for TNL_ASSERT_ALL_LT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_ALL_CMP_HELPER_( ALL_LT, less, < )
// Implements the helper function for TNL_ASSERT_ALL_GE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_ALL_CMP_HELPER_( ALL_GE, greaterEqual, >= )
// Implements the helper function for TNL_ASSERT_ALL_GT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_ALL_CMP_HELPER_( ALL_GT, greater, > )

   #undef TNL_IMPL_ALL_CMP_HELPER_

}  // namespace TNL::Assert

   #define TNL_ASSERT_ALL_LE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperALL_LE, <=, val1, val2, msg )
   #define TNL_ASSERT_ALL_LT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperALL_LT, <, val1, val2, msg )
   #define TNL_ASSERT_ALL_GE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperALL_GE, >=, val1, val2, msg )
   #define TNL_ASSERT_ALL_GT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperALL_GT, >, val1, val2, msg )

#endif  // #ifdef NDEBUG
