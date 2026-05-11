#pragma once

#include <limits>

#include <TNL/Arithmetics/experimental/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorHelperFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics::experimental;

// test fixture for typed tests
template< typename Vector >
class VectorTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using ViewType = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
};

// types for which VectorTest is instantiated
// TODO: Quad must be fixed
// Use diagonal selection instead of full Cartesian product to avoid
// combinatoric explosion. Each parameter dimension (ValueType, Device, IndexType)
// is fully covered, just not in all combinations.
using VectorTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   // Sequential + long index: representative ValueTypes
   Vector< int, Devices::Sequential, long >,
   Vector< double, Devices::Sequential, long >,
   //Vector< Quad< float >,  Devices::Sequential, long >,
   //Vector< Quad< double >, Devices::Sequential, long >,

   // Host + long index: all ValueTypes
   Vector< int, Devices::Host, long >,
   Vector< double, Devices::Host, long >,
   Vector< long, Devices::Host, long >,
   //Vector< Quad< float >,  Devices::Host, long >,
   //Vector< Quad< double >, Devices::Host, long >,

   // Host + non-long IndexTypes (covers short/int without repeating all ValueTypes)
   Vector< float, Devices::Host, short >,
   Vector< float, Devices::Host, int >
#elif defined( __CUDACC__ )
   // Same diagonal as Host portion, with Devices::Cuda
   Vector< int, Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >,
   Vector< long, Devices::Cuda, long >,
   //Vector< Quad< float >,  Devices::Cuda, long >,
   //Vector< Quad< double >, Devices::Cuda, long >,
   Vector< float, Devices::Cuda, short >,
   Vector< float, Devices::Cuda, int >
#elif defined( __HIP__ )
   // Same diagonal as Host portion, with Devices::Hip
   Vector< int, Devices::Hip, long >,
   Vector< double, Devices::Hip, long >,
   Vector< long, Devices::Hip, long >,
   //Vector< Quad< float >,  Devices::Hip, long >,
   //Vector< Quad< double >, Devices::Hip, long >,
   Vector< float, Devices::Hip, short >,
   Vector< float, Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( VectorTest, VectorTypes );
