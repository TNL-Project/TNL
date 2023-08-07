// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <TNL/Backend/Types.h>

#include "CudaSupportMissing.h"

namespace TNL::Exceptions {

class CudaRuntimeError : public std::runtime_error
{
public:
   CudaRuntimeError( Backend::error_t error_code )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code )
                         + "): " + description( error_code ) + "." ),
     code_( error_code )
   {}

   CudaRuntimeError( Backend::error_t error_code, const std::string& what_arg )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code )
                         + "): " + description( error_code ) + ".\nDetails: " + what_arg ),
     code_( error_code )
   {}

   CudaRuntimeError( Backend::error_t error_code, const char* file_name, int line )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code ) + "): "
                         + description( error_code ) + ".\nSource: line " + std::to_string( line ) + " in " + file_name ),
     code_( error_code )
   {}

   [[nodiscard]] Backend::error_t
   code() const
   {
      return code_;
   }

private:
   static std::string
   name( Backend::error_t error_code )
   {
#ifdef __CUDACC__
      return cudaGetErrorName( error_code );
#else
      throw CudaSupportMissing();
#endif
   }

   static std::string
   description( Backend::error_t error_code )
   {
#ifdef __CUDACC__
      return cudaGetErrorString( error_code );
#else
      throw CudaSupportMissing();
#endif
   }

   Backend::error_t code_;
};

}  // namespace TNL::Exceptions
