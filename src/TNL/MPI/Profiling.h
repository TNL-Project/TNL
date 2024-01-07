// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Timer.h>

namespace TNL::MPI {

[[nodiscard]] inline Timer&
getTimerAllreduce()
{
   static Timer t;
   return t;
}

}  // namespace TNL::MPI
