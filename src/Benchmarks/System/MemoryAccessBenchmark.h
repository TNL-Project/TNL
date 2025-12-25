// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

struct MemoryAccessBenchmark
{
   static void
   configSetup( TNL::Config::ConfigDescription& config );

   template< int ElementSize >
   static bool
   performBenchmark( const TNL::Config::ParameterContainer& parameters );
};

#include "MemoryAccessBenchmark.hpp"
