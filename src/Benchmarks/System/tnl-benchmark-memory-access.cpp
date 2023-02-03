// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/parseCommandLine.h>
#include "MemoryAccessBenchmark.h"

template< typename Device >
bool resolveElementSize( const TNL::Config::ParameterContainer& parameters )
{
   int element_size = parameters.getParameter< int >( "element-size" );
   switch( element_size )
   {
      case 1:
         return MemoryAccessBenchmark::performBenchmark< Device, 1 >( parameters );
      case 2:
         return MemoryAccessBenchmark::performBenchmark< Device, 2 >( parameters );
      case 4:
         return MemoryAccessBenchmark::performBenchmark< Device, 4 >( parameters );
      case 8:
         return MemoryAccessBenchmark::performBenchmark< Device, 8 >( parameters );
      case 16:
         return MemoryAccessBenchmark::performBenchmark< Device, 16 >( parameters );
      case 32:
         return MemoryAccessBenchmark::performBenchmark< Device, 32 >( parameters );
      case 64:
         return MemoryAccessBenchmark::performBenchmark< Device, 64 >( parameters );
      case 128:
         return MemoryAccessBenchmark::performBenchmark< Device, 128 >( parameters );
      case 256:
         return MemoryAccessBenchmark::performBenchmark< Device, 256 >( parameters );
   }
   std::cerr << "Element size " << element_size << " is not allowed. It can be only 1, 2, 4, 8, 16, 32, 64, 128, 256." << std::endl;
   return false;
}

int main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   MemoryAccessBenchmark::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( !parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   auto device = parameters.getParameter< TNL::String >( "device" );
   if( device == "sequential" )
      return resolveElementSize< TNL::Devices::Sequential >( parameters ) ? EXIT_SUCCESS : EXIT_FAILURE;
   if( device == "host" )
      return resolveElementSize< TNL::Devices::Host >( parameters ) ? EXIT_SUCCESS : EXIT_FAILURE;
   std::cerr << "Wrong device type " << device << " for the memory access benchmarking, only 'sequential' or 'host' is allowed here." << std::endl;
   return EXIT_FAILURE;
}
