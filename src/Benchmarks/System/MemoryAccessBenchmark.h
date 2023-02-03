// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#include <TNL/Config/ConfigDescription.h>

#pragma once

/**
 * \brief Benchmark for measuring efficiency of access of CPU to the system memory.
 *
 */
struct MemoryAccessBenchmark
{
   static void configSetup( TNL::Config::ConfigDescription& config );

   template< typename Device, int ElementSize >
   static bool performBenchmark( const TNL::Config::ParameterContainer& parameters );
};

#include "MemoryAccessBenchmark.hpp"