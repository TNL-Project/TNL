// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

//! \brief Namespace for TNL execution models
namespace TNL::Devices {

struct Sequential
{
   //! Not used by any sequential algorithm, only for compatibility with parallel execution models.
   struct LaunchConfiguration
   {};
};

}  // namespace TNL::Devices
