// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Containers/Array.h>

namespace TNL::Benchmarks {

template<
   typename Real = double,
   typename Index = int,
   template< typename > class HostAllocator = Allocators::Default< Devices::Host >::Allocator,
   template< typename > class CudaAllocator = Allocators::Default< Devices::GPU >::Allocator >
void
benchmarkArrayOperations( Benchmark& benchmark, const long& size )
{
   using HostArray = Containers::Array< Real, Devices::Host, Index, HostAllocator< Real > >;
   using CudaArray = Containers::Array< Real, Devices::GPU, Index, CudaAllocator< Real > >;

   std::size_t datasetSize = size * sizeof( Real );

   HostArray hostArray;
   HostArray hostArray2;
   CudaArray deviceArray;
   CudaArray deviceArray2;
   hostArray.setSize( size );
   hostArray2.setSize( size );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   deviceArray.setSize( size );
   deviceArray2.setSize( size );
#endif

   Real resultHost;

   // reset functions
   auto reset1 = [ & ]()
   {
      hostArray.setValue( 1.0 );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      deviceArray.setValue( 1.0 );
#endif
   };
   auto reset2 = [ & ]()
   {
      hostArray2.setValue( 1.0 );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      deviceArray2.setValue( 1.0 );
#endif
   };
   auto reset12 = [ & ]()
   {
      reset1();
      reset2();
   };

   reset12();

   if( std::is_fundamental_v< Real > ) {
      // std::memcmp
      auto compareHost = [ & ]()
      {
         resultHost = static_cast< Real >(
            std::memcmp( hostArray.getData(), hostArray2.getData(), hostArray.getSize() * sizeof( Real ) ) == 0 );
      };
      benchmark.setOperation( "comparison (memcmp)" );
      benchmark.setDatasetSize( 2 * datasetSize );
      benchmark.time< Devices::Host >( reset12, "TNL", compareHost );

      // std::memcpy and Backend::memcpy
      auto copyHost = [ & ]()
      {
         std::memcpy( hostArray.getData(), hostArray2.getData(), hostArray.getSize() * sizeof( Real ) );
      };
      benchmark.setOperation( "copy (memcpy)" );
      benchmark.setDatasetSize( 2 * datasetSize );
      benchmark.time< Devices::Host >( reset12, "TNL", copyHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      auto copyCuda = [ & ]()
      {
         Backend::memcpy(
            deviceArray.getData(),
            deviceArray2.getData(),
            deviceArray.getSize() * sizeof( Real ),
            Backend::MemcpyDeviceToDevice );
      };
      benchmark.time< Devices::GPU >( reset12, "TNL", copyCuda );
#endif
   }

   auto compareHost = [ & ]()
   {
      resultHost = static_cast< int >( hostArray == hostArray2 );
   };
   benchmark.setOperation( "comparison (operator==)" );
   benchmark.setDatasetSize( 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "TNL", compareHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Real resultDevice;
   auto compareCuda = [ & ]()
   {
      resultDevice = (int) ( deviceArray == deviceArray2 );
   };
   benchmark.time< Devices::GPU >( reset1, "TNL", compareCuda );
#endif

   auto copyAssignHostHost = [ & ]()
   {
      hostArray = hostArray2;
   };
   benchmark.setOperation( "copy (operator=)" );
   benchmark.setDatasetSize( 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "TNL", copyAssignHostHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   auto copyAssignCudaCuda = [ & ]()
   {
      deviceArray = deviceArray2;
   };
   benchmark.time< Devices::GPU >( reset1, "TNL", copyAssignCudaCuda );
#endif

#if defined( __CUDACC__ ) || defined( __HIP__ )
   auto copyAssignHostCuda = [ & ]()
   {
      deviceArray = hostArray;
   };
   auto copyAssignCudaHost = [ & ]()
   {
      hostArray = deviceArray;
   };
   benchmark.setOperation( "copy (operator=)" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Devices::GPU >( reset1, "host-to-device", copyAssignHostCuda );
   benchmark.time< Devices::GPU >( reset1, "device-to-host", copyAssignCudaHost );
#endif

   auto setValueHost = [ & ]()
   {
      hostArray.setValue( 3.0 );
   };
   benchmark.setOperation( "setValue" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Devices::Host >( reset1, "TNL", setValueHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   auto setValueCuda = [ & ]()
   {
      deviceArray.setValue( 3.0 );
   };
   benchmark.time< Devices::GPU >( reset1, "TNL", setValueCuda );
#endif

   auto setSizeHost = [ & ]()
   {
      hostArray.setSize( size );
   };
   auto resetSize1 = [ & ]()
   {
      hostArray.reset();
#if defined( __CUDACC__ ) || defined( __HIP__ )
      deviceArray.reset();
#endif
   };
   benchmark.setOperation( "allocation (setSize)" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Devices::Host >( resetSize1, "TNL", setSizeHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   auto setSizeCuda = [ & ]()
   {
      deviceArray.setSize( size );
   };
   benchmark.time< Devices::GPU >( resetSize1, "TNL", setSizeCuda );
#endif

   auto resetSizeHost = [ & ]()
   {
      hostArray.reset();
   };
   auto setSize1 = [ & ]()
   {
      hostArray.setSize( size );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      deviceArray.setSize( size );
#endif
   };
   benchmark.setOperation( "deallocation (reset)" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Devices::Host >( setSize1, "TNL", resetSizeHost );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   auto resetSizeCuda = [ & ]()
   {
      deviceArray.reset();
   };
   benchmark.time< Devices::GPU >( setSize1, "TNL", resetSizeCuda );
#endif
}

}  // namespace TNL::Benchmarks
