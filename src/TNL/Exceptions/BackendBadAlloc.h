// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <new>

#include <TNL/Backend/Types.h>  // includes HIP runtime headers

namespace TNL::Exceptions {

struct BackendBadAlloc : public std::bad_alloc
{
   BackendBadAlloc()  // NOLINT
   {
      // Make sure to clear the CUDA/HIP error, otherwise the exception
      // handler might throw another exception with the same error.
#if defined( __CUDACC__ )
      (void) cudaGetLastError();
#elif defined( __HIP__ )
      (void) hipGetLastError();
#endif
   }

   [[nodiscard]] const char*
   what() const noexcept override
   {
      return "The device backend failed to allocate memory: "
             "most likely there is not enough space on the device.";
   }
};

}  // namespace TNL::Exceptions
