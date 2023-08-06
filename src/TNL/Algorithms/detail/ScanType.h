// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::detail {

enum class ScanType
{
   Exclusive,
   Inclusive
};

enum class ScanPhaseType
{
   WriteInFirstPhase,
   WriteInSecondPhase
};

}  // namespace TNL::Algorithms::detail
