#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Ralston4.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Ralston4<>;

constexpr double expected_eoc = 4.1;

#include "ODEStaticSolverTest.h"
