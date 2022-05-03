
#define KERNEL KERNEL_VALUE
#define DIMENSION DIMENSION_VALUE

#include KERNEL
#include "../support/ImageSolver.h"

#include <TNL/Config/parseCommandLine.h>

using TaskSolver = ImageSolver;

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
