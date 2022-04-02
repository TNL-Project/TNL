
#include "../kernels/naive.h"
#include "../support/DummyBenchmark.h"

#include <TNL/Config/parseCommandLine.h>

#define DIMENSION DIMENSION_VALUE

using TaskBenchmark = DummyBenchmark< DIMENSION, TNL::Devices::Cuda >;

int main(int argc, char* argv[])
{
   TaskBenchmark benchmark;

   auto config = benchmark.makeInputConfig();

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   benchmark.run( parameters );

   return 0;
}
