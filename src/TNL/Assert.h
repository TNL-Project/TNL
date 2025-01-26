// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <stdexcept>

/**
 * \file Assert.h
 *
 * \brief The purpose of this file is to define the `TNL_ASSERT_*` debugging
 * macros.
 *
 * If the `NDEBUG` macro is defined, the build is considered to be optimized
 * and all assert macros are empty. Otherwise, the conditions are checked and
 * failures lead to the `TNL::Assert::AssertionError` exception containing
 * the diagnostics message.
 */

//! \brief Internal namespace for helper classes used in the `TNL_ASSERT_*` macros.
namespace TNL::Assert {

//! \brief Exception that represents an assertion error and its diagnostics.
struct AssertionError : public std::runtime_error
{
   AssertionError( const std::string& msg )
   : std::runtime_error( msg )
   {}
};

}  // namespace TNL::Assert

// check the minimum version of the C++ standard required by TNL, otherwise
// provide a useful error message for each supported compiler/platform
#if __cplusplus < 201703L
   #if defined( __clang__ ) || defined( __GNUC__ ) || defined( __GNUG__ )
      #error "TNL requires the C++17 standard or later. Did you forget to specify the -std=c++17 compiler option?"
   #elif defined( _MSC_VER )
      #error "TNL requires the C++17 standard or later. Did you forget to specify the /std:c++17 compiler option?"
   #else
      #error "TNL requires the C++17 standard or later. Make sure it is enabled in your compiler options."
   #endif
#endif

#if defined( __NVCC__ )
   // check for required compiler features and provide a useful error message
   #if ! defined( __CUDACC_RELAXED_CONSTEXPR__ ) || ! defined( __CUDACC_EXTENDED_LAMBDA__ )
      #error "TNL requires the following flags to be specified for nvcc: --expt-relaxed-constexpr --extended-lambda"
   #endif
#endif

#ifdef NDEBUG

   // empty macros for optimized build
   /**
    * \brief Asserts that the expression \e val evaluates to \e true.
    *
    * The assertion succeeds if, and only if, \e val evaluates to equal to \e true.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_TRUE( val, msg )
   /**
    * \brief Asserts that the expression \e val evaluates to \e false.
    *
    * The assertion succeeds if, and only if, \e val evaluates to equal to \e false.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_FALSE( val, msg )
   /**
    * \brief Asserts that the expression \e val1 is equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 and \e val2 are equal.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_EQ( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is not equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 and \e val2 are not equal.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_NE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is less than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is less than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_LE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is less than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is less than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_LT( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is greater than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is greater than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_GE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is greater than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is greater than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_GT( val1, val2, msg )

#else  // #ifdef NDEBUG

   #include <iostream>
   #include <sstream>
   #include <cstdio>

   #include <TNL/Backend/Macros.h>

   // reference: https://github.com/ROCm-Developer-Tools/HIP/issues/2235
   #if defined( __HIP__ ) && ! defined( HIP_ENABLE_PRINTF )
      #error \
         "TNL requires the HIP_ENABLE_PRINTF macro to be defined in debug mode in order to enable assert messages frorm HIP device kernels."
   #endif

namespace TNL::Assert {

inline void
abortWithDiagnosticsHost( const char* assertion,
                          const char* message,
                          const char* file,
                          const char* function,
                          int line,
                          const char* diagnostics )
{
   std::stringstream str;
   str << "Assertion '" << assertion << "' failed !!!\n"
       << "Message: " << message << "\n"
       << "File: " << file << "\n"
       << "Function: " << function << "\n"
       << "Line: " << line << "\n"
       << "Diagnostics:\n"
       << diagnostics << std::endl;
   throw AssertionError( str.str() );
}

__cuda_callable__
inline void
abortWithDiagnosticsCuda( const char* assertion,
                          const char* message,
                          const char* file,
                          const char* function,
                          int line,
                          const char* diagnostics )
{
   // NOTE: HIP requires printf instead of std::printf (the latter is not __host__ __device__)
   // FIXME: using printf in HIP kernels hangs on gfx803
   #if ! defined( __HIP_DEVICE_COMPILE__ )
   std::printf( "Assertion '%s' failed !!!\n"
                "Message: %s\n"
                "File: %s\n"
                "Function: %s\n"
                "Line: %d\n"
                "Diagnostics: %s\n",
                assertion,
                message,
                file,
                function,
                line,
                diagnostics );
   #endif

   #ifdef __CUDA_ARCH__
   // https://devtalk.nvidia.com/default/topic/509584/how-to-cancel-a-running-cuda-kernel-/
   // it is reported as "illegal instruction", but that leads to an abort as well...
   asm( "trap;" );
   #elif defined __HIP_DEVICE_COMPILE__
   abort();  // FIXME: unlike CUDA, the ROCm runtime aborts the whole program rather than just the kernel
   #endif
}

template< typename T >
struct Formatter
{
   static std::string
   printToString( const T& value )
   {
      std::stringstream ss;
      ss << value;
      return ss.str();
   }
};

template<>
struct Formatter< bool >
{
   static std::string
   printToString( const bool& value )
   {
      if( value )
         return "true";
      else
         return "false";
   }
};

template< typename T, typename U >
struct Formatter< std::pair< T, U > >
{
   static std::string
   printToString( const std::pair< T, U >& pair )
   {
      std::stringstream ss;
      ss << '(' << pair.first << ',' << pair.second << ')';
      return ss.str();
   }
};

template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperOpFailure( const char* assertion,
                    const char* message,
                    const char* file,
                    const char* function,
                    int line,
                    const char* lhs_expression,
                    const char* rhs_expression,
                    const T1& lhs_value,
                    const T2& rhs_value,
                    const char* op )
{
   #if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   // diagnostics is not supported - we don't have the machinery
   // to construct the dynamic error message
   abortWithDiagnosticsCuda( assertion, message, file, function, line, "Not supported in CUDA kernels." );
   #else
   const std::string formatted_lhs_value = Formatter< T1 >::printToString( lhs_value );
   const std::string formatted_rhs_value = Formatter< T2 >::printToString( rhs_value );
   std::stringstream str;
   if( std::string( op ) == "==" ) {
      str << "      Expected: " << lhs_expression;
      if( formatted_lhs_value != lhs_expression ) {
         str << "\n      Which is: " << formatted_lhs_value;
      }
      str << "\nTo be equal to: " << rhs_expression;
      if( formatted_rhs_value != rhs_expression ) {
         str << "\n      Which is: " << formatted_rhs_value;
      }
      str << std::endl;
   }
   else {
      str << "Expected: (" << lhs_expression << ") " << op << " (" << rhs_expression << "), "
          << "actual: " << formatted_lhs_value << " vs " << formatted_rhs_value << std::endl;
   }
   abortWithDiagnosticsHost( assertion, message, file, function, line, str.str().c_str() );
   #endif
}

TNL_NVCC_HD_WARNING_DISABLE
template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperTrue( const char* assertion,
               const char* message,
               const char* file,
               const char* function,
               int line,
               const char* expr1,
               const char* expr2,
               const T1& val1,
               const T2& val2 )
{
   // explicit cast is necessary, because T1::operator! might not be defined
   if( ! (bool) val1 )
      ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, "true", val1, true, "==" );
}

TNL_NVCC_HD_WARNING_DISABLE
template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperFalse( const char* assertion,
                const char* message,
                const char* file,
                const char* function,
                int line,
                const char* expr1,
                const char* expr2,
                const T1& val1,
                const T2& val2 )
{
   if( val1 )
      ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, "false", val1, false, "==" );
}

   // A macro for implementing the helper functions needed to implement
   // TNL_ASSERT_??. It is here just to avoid copy-and-paste of similar code.
   #define TNL_IMPL_CMP_HELPER_( op_name, op )                                                                            \
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
         if( ! ( (val1) op( val2 ) ) )                                                                                    \
            ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, expr2, val1, val2, #op ); \
      }

// Implements the helper function for TNL_ASSERT_EQ
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( EQ, == )
// Implements the helper function for TNL_ASSERT_NE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( NE, != )
// Implements the helper function for TNL_ASSERT_LE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( LE, <= )
// Implements the helper function for TNL_ASSERT_LT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( LT, < )
// Implements the helper function for TNL_ASSERT_GE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( GE, >= )
// Implements the helper function for TNL_ASSERT_GT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( GT, > )

   #undef TNL_IMPL_CMP_HELPER_

}  // namespace TNL::Assert

   // Internal macro wrapping the __PRETTY_FUNCTION__ "magic".
   #if defined( _MSC_VER )
      #define __TNL_PRETTY_FUNCTION __FUNCSIG__
   #else
      #define __TNL_PRETTY_FUNCTION __PRETTY_FUNCTION__
   #endif

   // On Linux, __STRING is defined in glibc's sys/cdefs.h, but there is no such
   // header on Windows and possibly other platforms.
   #ifndef __STRING
      #define __STRING( arg ) #arg
   #endif

   // Internal macro to compose the string representing the assertion.
   // We can't do it easily at runtime, because we have to support assertions
   // in CUDA kernels, which can't use std::string objects. Instead, we do it
   // at compile time - adjacent strings are joined at the language level.
   #define __TNL_JOIN_STRINGS( val1, op, val2 ) __STRING( val1 ) " " __STRING( op ) " " __STRING( val2 )

   // Internal macro to pass all the arguments to the specified cmpHelperOP
   #define __TNL_ASSERT_PRED2( pred, op, val1, val2, msg ) \
      pred( __TNL_JOIN_STRINGS( val1, op, val2 ), msg, __FILE__, __TNL_PRETTY_FUNCTION, __LINE__, #val1, #val2, val1, val2 )

   // Main definitions of the TNL_ASSERT_* macros
   // unary
   #define TNL_ASSERT_TRUE( val, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperTrue, ==, val, true, msg )
   #define TNL_ASSERT_FALSE( val, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperFalse, ==, val, false, msg )
   // binary
   #define TNL_ASSERT_EQ( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperEQ, ==, val1, val2, msg )
   #define TNL_ASSERT_NE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperNE, !=, val1, val2, msg )
   #define TNL_ASSERT_LE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperLE, <=, val1, val2, msg )
   #define TNL_ASSERT_LT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperLT, <, val1, val2, msg )
   #define TNL_ASSERT_GE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperGE, >=, val1, val2, msg )
   #define TNL_ASSERT_GT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperGT, >, val1, val2, msg )

#endif  // #ifdef NDEBUG
