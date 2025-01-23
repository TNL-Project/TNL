// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string_view>

/**
 * \brief Namespace for TNL exceptions.
 */
namespace TNL::Exceptions {

struct BackendSupportMissing : public std::runtime_error
{
   BackendSupportMissing()
   : std::runtime_error( message.data() )
   {}

private:
   static constexpr std::string_view message =  //
      "Support for parallel computing backend is missing, but the program "
      "called a function which needs it. Please recompile the program with "
      "support for the desired backend (CUDA or HIP).";
};

}  // namespace TNL::Exceptions
