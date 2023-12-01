#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Kutta.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Kutta<>;

constexpr double expected_eoc = 4.0;  // Kutta is a 3th order method but the test reports EOC=4.0

#include "ODEStaticSolverTest.h"
