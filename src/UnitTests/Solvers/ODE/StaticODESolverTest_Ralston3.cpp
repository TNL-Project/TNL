#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Ralston3.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Ralston3<>;

constexpr double expected_eoc = 3.0;

#include "ODEStaticSolverTest.h"
