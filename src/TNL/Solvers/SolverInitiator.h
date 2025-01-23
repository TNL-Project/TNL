// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL::Solvers {

template< template< typename Real,
                    typename Device,
                    typename Index,
                    typename MeshType,
                    typename ConfigTag,
                    typename SolverStarter > class ProblemSetter,
          typename ConfigTag >
class SolverInitiator
{
public:
   static bool
   run( const Config::ParameterContainer& parameters );
};

}  // namespace TNL::Solvers

#include <TNL/Solvers/SolverInitiator.hpp>
