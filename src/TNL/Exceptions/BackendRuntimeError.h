// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>

#include <TNL/Backend/Types.h>

#include "BackendSupportMissing.h"

namespace TNL::Exceptions {

class BackendRuntimeError : public std::runtime_error
{
public:
   BackendRuntimeError( Backend::error_t error_code )
   : std::runtime_error( message_prefix.data() + code_string( error_code ) + " (" + name( error_code )
                         + "): " + description( error_code ) + "." ),
     code_( error_code )
   {
      clear_error();
   }

   BackendRuntimeError( Backend::error_t error_code, const std::string& what_arg )
   : std::runtime_error( message_prefix.data() + code_string( error_code ) + " (" + name( error_code )
                         + "): " + description( error_code ) + ".\nDetails: " + what_arg ),
     code_( error_code )
   {
      clear_error();
   }

   BackendRuntimeError( Backend::error_t error_code, const char* file_name, int line )
   : std::runtime_error( message_prefix.data() + code_string( error_code ) + " (" + name( error_code ) + "): "
                         + description( error_code ) + ".\nSource: line " + std::to_string( line ) + " in " + file_name ),
     code_( error_code )
   {
      clear_error();
   }

   [[nodiscard]] Backend::error_t
   code() const
   {
      return code_;
   }

private:
#if defined( __CUDACC__ )
   static constexpr std::string_view message_prefix = "CUDA ERROR ";
#elif defined( __HIP__ )
   static constexpr std::string_view message_prefix = "HIP ERROR ";
#else
   static constexpr std::string_view message_prefix = "ERROR ";
#endif

   static void
   clear_error()
   {
      // Make sure to clear the CUDA/HIP error, otherwise the exception
      // handler might throw another exception with the same error.
#if defined( __CUDACC__ )
      (void) cudaGetLastError();
#elif defined( __HIP__ )
      (void) hipGetLastError();
#endif
   }

   static std::string
   code_string( Backend::error_t error_code )
   {
      return std::to_string( static_cast< int >( error_code ) );
   }

   static std::string
   name( Backend::error_t error_code )
   {
#if defined( __CUDACC__ )
      return cudaGetErrorName( error_code );
#elif defined( __HIP__ )
      return hipGetErrorName( error_code );
#else
      throw BackendSupportMissing();
#endif
   }

   static std::string
   description( Backend::error_t error_code )
   {
#if defined( __CUDACC__ )
      return cudaGetErrorString( error_code );
#elif defined( __HIP__ )
      return hipGetErrorString( error_code );
#else
      throw BackendSupportMissing();
#endif
   }

   Backend::error_t code_;
};

}  // namespace TNL::Exceptions
