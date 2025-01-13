#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

//#if defined( __CUDACC__ ) || defined( __HIP__ )
#include <thrust/execution_policy.h>
//#endif

namespace TNL {
namespace Thrust {

//#ifdef __CUDACC__

template< typename DeviceType >
struct ThrustExecutionPolicySelector;

template<>
struct ThrustExecutionPolicySelector< TNL::Devices::Host >
{
   using type = typename thrust::detail::host_t;
};

template<>
struct ThrustExecutionPolicySelector< TNL::Devices::Cuda >
{
   using type = typename thrust::detail::device_t;
};

template< typename DeviceType >
using ThrustExecutionPolicy = typename ThrustExecutionPolicySelector< DeviceType >::type;

//#endif

} // Thrust
} // TNL

