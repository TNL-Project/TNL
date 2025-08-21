// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>

#include "dense-linear-solvers.h"

void
setupConfig( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark setting:" );
   config.addEntry< int >( "matrix-size", "Size of the randomly generated matrix.", 128 );
   config.addEntry< TNL::String >( "input-file", "Input matrix file name (overrides random matrix generation)." );
   config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-dense-linear-solvers.log" );
   config.addEntry< bool >( "append-log", "Append to log file.", false );

   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< TNL::String >( "device", "Device to compute on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< TNL::String >( "pivoting", "Use pivoting in GEM/LU computation.", "yes" );
   config.addEntryEnum( "yes" );
   config.addEntryEnum( "no" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

void
resolvePrecision( TNL::Config::ParameterContainer& parameters )
{
   TNL::String precision = parameters.getParameter< TNL::String >( "precision" );
   if( precision == "float" || precision == "all" ) {
      benchmarkDenseLinearSolvers< float, int >( parameters );
   }
   if( precision == "double" || precision == "all" ) {
      benchmarkDenseLinearSolvers< double, int >( parameters );
   }
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   resolvePrecision( parameters );
   return EXIT_SUCCESS;
}
