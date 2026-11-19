// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL::Benchmarks {

/**
 * \brief Returns a lowercase, backend-specific name for the given device type.
 *
 * The returned strings match the CLI `--device` parameter values used across
 * benchmarks:
 *
 * | Device type              | Return value  |
 * |--------------------------|---------------|
 * | `Devices::Sequential`    | `"sequential"`|
 * | `Devices::Host`          | `"host"`      |
 * | `Devices::GPU` (CUDA)    | `"cuda"`      |
 * | `Devices::GPU` (HIP)     | `"hip"`       |
 * | `Devices::GPU` (neither) | `"gpu"`       |
 * | anything else            | `"unknown"`   |
 *
 * \tparam Device A device type from `TNL::Devices` (Sequential, Host, GPU)
 */
template< typename Device >
std::string
getDeviceName()
{
   if constexpr( std::is_same_v< Device, Devices::Sequential > ) {
      return "sequential";
   }
   else if constexpr( std::is_same_v< Device, Devices::Host > ) {
      return "host";
   }
   else if constexpr( std::is_same_v< Device, Devices::GPU > ) {
#if defined( __CUDACC__ )
      return "cuda";
#elif defined( __HIP__ )
      return "hip";
#else
      return "gpu";
#endif
   }
   else {
      return "unknown";
   }
}

/**
 * \brief Checks whether a device type should be active for the given CLI parameters.
 *
 * Inspects the `"device"` parameter from the configuration and returns `true`
 * if the specified `Device` type matches the user's selection. The special
 * value `"all"` activates all supported devices.
 *
 * Recognized parameter values:
 * - `"all"`        — activates every device type
 * - `"sequential"` — activates `Devices::Sequential`
 * - `"host"`       — activates `Devices::Host`
 * - `"cuda"`       — activates `Devices::GPU` (when built with CUDA)
 * - `"hip"`        — activates `Devices::GPU` (when built with HIP)
 *
 * \tparam Device A device type from `TNL::Devices`
 * \param parameters Parsed CLI configuration parameters
 * \return `true` if this device should run, `false` otherwise
 */
template< typename Device >
bool
checkDevice( const Config::ParameterContainer& parameters )
{
   const auto device = parameters.getParameter< std::string >( "device" );
   if( device == "all" )
      return true;
   if constexpr( std::is_same_v< Device, Devices::Sequential > )
      return device == "sequential";
   else if constexpr( std::is_same_v< Device, Devices::Host > )
      return device == "host";
   else if constexpr( std::is_same_v< Device, Devices::GPU > )
      return device == "cuda" || device == "hip";
   else
      return false;
}

/**
 * \brief Returns the launch configuration of the given device
 * with which the host does not wait for the kernels to finish.
 *
 * On GPUs, the kernels launched with this configuration in the same stream
 * run in the order of their launches,
 * but the host has to synchronize the device before it reads the results.
 * The time measurement by \ref Benchmark::time synchronizes the device
 * after each measured function.
 * The configurations of the other devices are the default ones.
 *
 * \tparam Device A device type from `TNL::Devices`
 * \return The launch configuration.
 */
template< typename Device >
typename Device::LaunchConfiguration
getAsyncLaunchConfiguration()
{
   typename Device::LaunchConfiguration launchConfiguration;
   if constexpr( std::is_same_v< Device, Devices::GPU > )
      launchConfiguration.blockHostUntilFinished = false;
   return launchConfiguration;
}

}  // namespace TNL::Benchmarks
