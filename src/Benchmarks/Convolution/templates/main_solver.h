
#include "../kernels/naive.h"
#include "../support/DummySolver.h"

#include <TNL/Config/parseCommandLine.h>

#define DIMENSION DIMENSION_VALUE

using TaskSolver = DummySolver< DIMENSION, TNL::Devices::Cuda >;

int main(int argc, char* argv[])
{
   TaskSolver solver;

   auto config = solver.makeInputConfig();

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   solver.solve( parameters );

   return 0;
}
