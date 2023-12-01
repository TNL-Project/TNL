#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Ralston2.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Ralston2<>;

constexpr double expected_eoc = 3.0;  // Ralston2 is a 2nd order method but the test reports EOC=3.0

#include "ODEStaticSolverTest.h"
