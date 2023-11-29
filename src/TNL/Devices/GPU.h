// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/KernelLaunch.h>

namespace TNL::Devices {

class GPU
{
public:
   //! \brief Alias to the GPU kernel launch configuration structure.
   using LaunchConfiguration = TNL::Backend::LaunchConfiguration;

   static inline void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" )
   {
#if defined( __HIP__ )
      const char* message = "Choose HIP device to run the computation.";
      config.addEntry< int >( prefix + "hip-device", message, 0 );
#elif defined( __CUDACC__ )
      const char* message = "Choose CUDA device to run the computation.";
      config.addEntry< int >( prefix + "cuda-device", message, 0 );
#else
      const char* message = "Choose CUDA device to run the computation (not supported on this system).";
      config.addEntry< int >( prefix + "cuda-device", message, 0 );
#endif
   }

   static inline bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" )
   {
#if defined( __CUDACC__ ) || defined( __HIP__ )
   #if defined( __HIP__ )
      const char* name = "hip-device";
   #else
      const char* name = "cuda-device";
   #endif
      const int cudaDevice = parameters.getParameter< int >( prefix + name );
      Backend::setDevice( cudaDevice );
#endif
      return true;
   }
};

}  // namespace TNL::Devices
