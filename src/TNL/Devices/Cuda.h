// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GPU.h"

namespace TNL::Devices {

/**
 * \brief An alias to \ref GPU for convenience.
 *
 * It is not possible to build for multiple GPU backends at the same time, so
 * we can alias the types and avoid a huge amount of code duplication.
 */
using Cuda = GPU;

}  // namespace TNL::Devices
