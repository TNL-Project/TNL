#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Rule38.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Rule38<>;

constexpr double expected_eoc = 4.0;

#include "ODEStaticSolverTest.h"
