#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/SSPRK3.h>

using ODEMethod = TNL::Solvers::ODE::Methods::SSPRK3<>;

constexpr double expected_eoc = 4.0;  // SSPRK3 is a 3th order method but the test reports EOC=4.0

#include "ODEStaticSolverTest.h"
