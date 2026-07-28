// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/**
 * \brief Convenience header file which includes all headers from the
 `TNL/Backend/` subdirectory.
 *
 * Users may use this to avoid having to include many header files in their projects.
 * On the other hand,
 * parts of the TNL library should generally include only the specific headers they need,
 * in order to avoid cycles in the header inclusion.
 */

#include <TNL/Backend/Types.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/Stream.h>
#include <TNL/Backend/StreamPool.h>
#include <TNL/Backend/DeviceInfo.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Backend/KernelLaunch.h>
