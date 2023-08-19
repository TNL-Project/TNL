// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/KernelLaunch.h>

namespace TNL::Devices {

class Cuda
{
public:
   //! \brief Alias to the CUDA kernel launch configuration structure.
   using LaunchConfiguration = TNL::Backend::LaunchConfiguration;

   static inline void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" )
   {
#ifdef __CUDACC__
      const char* message = "Choose CUDA device to run the computation.";
#else
      const char* message = "Choose CUDA device to run the computation (not supported on this system).";
#endif
      config.addEntry< int >( prefix + "cuda-device", message, 0 );
   }

   static inline bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" )
   {
      const int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
      Backend::setDevice( cudaDevice );
      return true;
   }
};

}  // namespace TNL::Devices
