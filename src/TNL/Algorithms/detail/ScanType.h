// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace TNL::Algorithms::detail {

enum class ScanType : std::uint8_t
{
   Exclusive,
   Inclusive
};

enum class ScanPhaseType : std::uint8_t
{
   WriteInFirstPhase,
   WriteInSecondPhase
};

}  // namespace TNL::Algorithms::detail
