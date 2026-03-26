#pragma once

#ifdef HAVE_CUDA_SAMPLES
   #include <2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu>
#endif
#include <TNL/Containers/Array.h>

namespace TNL {

/**
 * Bitonic sort from NVIDIA CUDA samples.
 *
 * Implemented only for `unsigned int` value type and power-of-two sizes.
 */
struct NvidiaBitonicSort
{
   template< typename Array >
   static void
   sort( Array& array )
   {
#ifdef HAVE_CUDA_SAMPLES
      using ValueType = typename Array::ValueType;
      using DeviceType = typename Array::DeviceType;
      using IndexType = typename Array::IndexType;

      static_assert( std::is_same_v< DeviceType, Devices::Cuda >, "NvidiaBitonicSort requires Devices::Cuda" );
      static_assert( std::is_same_v< ValueType, unsigned int >, "NvidiaBitonicSort requires unsigned int value type" );

      const auto size = array.getSize();

      // Only power-of-two array lengths are supported by this implementation
      if( ! TNL::isPow2( size ) )
         throw std::runtime_error( "NvidiaBitonicSort supports only power-of-two array lengths" );

      // The NVIDIA bitonic sort works on key-value pairs so we need a second array.
      Containers::Array< ValueType, DeviceType, IndexType > valArray;
      valArray.setSize( size );

      ::bitonicSort(
         array.getData(),     // DstKey
         valArray.getData(),  // DstVal
         array.getData(),     // SrcKey
         valArray.getData(),  // SrcVal
         1,                   // batchSize
         size,                // arrayLength
         1 );                 // dir (ascending)
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );
#endif
   }
};

}  // namespace TNL
