// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <new>
#include <string>

#include <TNL/Backend/Types.h>  // includes HIP runtime headers

namespace TNL::Exceptions {

struct BackendBadAlloc : public std::bad_alloc
{
   explicit BackendBadAlloc( std::size_t size )
   : size_( size ),
     code_( read_error() )
   {
      message_ = "The device backend failed to allocate " + std::to_string( size ) + " bytes" + error_details( code_ );
   }

   // size of the failed allocation in bytes
   [[nodiscard]] std::size_t
   size() const noexcept
   {
      return size_;
   }

   // error reported by the backend, or zero if no error was pending
   [[nodiscard]] Backend::error_t
   code() const noexcept
   {
      return code_;
   }

   [[nodiscard]] const char*
   what() const noexcept override
   {
      return message_.c_str();
   }

private:
   // The error is read and cleared at the same time, so that the exception handler cannot be
   // confused by an error left behind by the failed allocation.
   [[nodiscard]] static Backend::error_t
   read_error()
   {
#if defined( __CUDACC__ )
      return cudaGetLastError();
#elif defined( __HIP__ )
      return hipGetLastError();
#else
      return 0;
#endif
   }

   [[nodiscard]] static std::string
   error_details( Backend::error_t error_code )
   {
#if defined( __CUDACC__ )
      if( error_code != cudaSuccess )
         return std::string( ": CUDA ERROR " ) + std::to_string( static_cast< int >( error_code ) ) + " ("
              + cudaGetErrorName( error_code ) + "): " + cudaGetErrorString( error_code ) + ".";
#elif defined( __HIP__ )
      if( error_code != hipSuccess )
         return std::string( ": HIP ERROR " ) + std::to_string( static_cast< int >( error_code ) ) + " ("
              + hipGetErrorName( error_code ) + "): " + hipGetErrorString( error_code ) + ".";
#else
      (void) error_code;
#endif
      return ": most likely there is not enough space on the device.";
   }

   std::string message_;
   std::size_t size_;
   Backend::error_t code_;
};

}  // namespace TNL::Exceptions
