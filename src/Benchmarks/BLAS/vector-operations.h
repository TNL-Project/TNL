// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>      // srand48
#include <type_traits>  // std::is_same_v
#include <algorithm>    // std::max_element, std::min_element, std::transform, etc.
#include <numeric>      // std::reduce, std::transform_reduce
#include <functional>   // std::function

#include <TNL/Benchmarks/Benchmark.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/LinearCombination.h>
#include "CommonVectorOperations.h"
#include "VectorOperations.h"
#include "../ExecutionPolicies.h"

#ifdef HAVE_BLAS
   #include "blasWrappers.h"
#endif

#if defined( __CUDACC__ )
   #include "cublasWrappers.h"
#elif defined( __HIP__ )
   #include "hipblasWrappers.h"
#endif

#ifdef HAVE_THRUST
   #include <thrust/reduce.h>
   #include <thrust/transform_reduce.h>
   #include <thrust/inner_product.h>
   #include <thrust/extrema.h>
   #include <thrust/execution_policy.h>
   #include <thrust/device_ptr.h>
#endif

namespace TNL::Benchmarks {

template< typename Real = double >
struct TwoVectorsCoefficients
{
   static constexpr std::size_t
   getSize() noexcept
   {
      return coefficients.size();
   }

   static constexpr Real
   getValue( const std::size_t i ) noexcept
   {
      return coefficients[ i ];
   }
   static constexpr std::array< Real, 2 > coefficients = { 1.0, 1.0 };
};

template< typename Real = double >
struct ThreeVectorsCoefficients
{
   static constexpr std::size_t
   getSize() noexcept
   {
      return coefficients.size();
   }

   static constexpr Real
   getValue( const std::size_t i ) noexcept
   {
      return coefficients[ i ];
   }
   static constexpr std::array< Real, 3 > coefficients = { 1.0, 1.0, 1.0 };
};

template< typename Device, typename Real = double, typename Index = int >
class VectorOperationsBenchmark
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector = Containers::Vector< Real, Devices::GPU, Index >;
   using SequentialView = Containers::VectorView< Real, Devices::Sequential, Index >;
   using HostView = Containers::VectorView< Real, Devices::Host, Index >;
   using CudaView = Containers::VectorView< Real, Devices::GPU, Index >;

   Benchmark& benchmark;
   long size = 0;
   std::size_t datasetSize = 0;

   HostVector hostVector;
   HostVector hostVector2;
   HostVector hostVector3;
   HostVector hostVector4;
   CudaVector deviceVector;
   CudaVector deviceVector2;
   CudaVector deviceVector3;
   CudaVector deviceVector4;

   HostView hostView;
   HostView hostView2;
   HostView hostView3;
   HostView hostView4;
   CudaView deviceView;
   CudaView deviceView2;
   CudaView deviceView3;
   CudaView deviceView4;

   Real resultHost;
   Real resultDevice;

   // reset functions
   std::function< void() > reset1;
   std::function< void() > reset2;
   std::function< void() > reset3;
   std::function< void() > reset4;
   std::function< void() > resetAll;

#if defined( __CUDACC__ )
   cublasHandle_t cublasHandle;
   const char* gpuBlasName = "cuBLAS";
#elif defined( __HIP__ )
   hipblasHandle_t hipblasHandle;
   const char* gpuBlasName = "hipBLAS";
#else
   // used only in the GPU branches, which are not instantiated in host-only builds
   const char* gpuBlasName = "";
#endif

public:
   VectorOperationsBenchmark( Benchmark& benchmark, const long& size )
   : benchmark( benchmark ),
     size( size ),
     datasetSize( size * sizeof( Real ) )
   {
      hostVector.setSize( size );
      hostVector2.setSize( size );
      hostVector3.setSize( size );
      hostVector4.setSize( size );
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         deviceVector.setSize( size );
         deviceVector2.setSize( size );
         deviceVector3.setSize( size );
         deviceVector4.setSize( size );
      }

      hostView.bind( hostVector );
      hostView2.bind( hostVector2 );
      hostView3.bind( hostVector3 );
      hostView4.bind( hostVector4 );
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         deviceView.bind( deviceVector );
         deviceView2.bind( deviceVector2 );
         deviceView3.bind( deviceVector3 );
         deviceView4.bind( deviceVector4 );
      }

      // reset functions
      // (Make sure to always use some in benchmarks, even if it's not necessary
      // to assure correct result - it helps to clear cache and avoid optimizations
      // of the benchmark loop.)
      reset1 = [ & ]()
      {
         hostVector.setValue( 1.0 );
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            deviceVector.setValue( 1.0 );
         // A relatively harmless call to keep the compiler from realizing we
         // don't actually do any useful work with the result of the reduction.
         srand( static_cast< unsigned int >( resultHost ) );
         resultHost = resultDevice = 0.0;
      };
      reset2 = [ & ]()
      {
         hostVector2.setValue( 1.0 );
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            deviceVector2.setValue( 1.0 );
      };
      reset3 = [ & ]()
      {
         hostVector3.setValue( 1.0 );
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            deviceVector3.setValue( 1.0 );
      };
      reset4 = [ & ]()
      {
         hostVector4.setValue( 1.0 );
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            deviceVector4.setValue( 1.0 );
      };

      resetAll = [ & ]()
      {
         reset1();
         reset2();
         reset3();
         reset4();
      };

      resetAll();

#if defined( __CUDACC__ )
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         cublasCreate( &cublasHandle );
#elif defined( __HIP__ )
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         hipblasCreate( &hipblasHandle );
#endif
   }

   ~VectorOperationsBenchmark()
   {
#if defined( __CUDACC__ )
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         cublasDestroy( cublasHandle );
#elif defined( __HIP__ )
      if constexpr( std::is_same_v< Device, Devices::GPU > )
         hipblasDestroy( hipblasHandle );
#endif
   }

   // verifying the results is useful to ensure that we compute the same thing
   // even with the 3rdparty libraries, where the compute functions may be more
   // complicated than just one function call
   void
   verify( const std::string& performer, Real result, Real expected, Real tolerance = 1e-6 )
   {
      if( TNL::abs( result - expected ) > tolerance ) {
         std::cerr << "ERROR: result " << result << " computed by " << performer << " is not equal to the expected value "
                   << expected << '\n';
      }
   }

   void
   verify( const std::string& performer, const HostVector& result, Real expected, Real tolerance = 1e-6 )
   {
      for( Index i = 0; i < size; i++ ) {
         if( TNL::abs( result[ i ] - expected ) > tolerance ) {
            std::cerr << "ERROR: " << i << "-th value of the result vector " << result[ i ] << " computed by " << performer
                      << " is not equal to the expected value " << expected << '\n';
         }
      }
   }

   void
   verify( const std::string& performer, const CudaVector& result, Real expected, Real tolerance = 1e-6 )
   {
      hostVector = result;
      verify( performer, hostVector, expected, tolerance );
      reset1();
   }

   void
   max()
   {
      benchmark.setOperation( "max" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMax( hostVector );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, 1.0 );

         auto computeET = [ & ]()
         {
            using TNL::max;
            resultHost = max( hostView );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, 1.0 );
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = *std::max_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::max_element", computeSTL );
         verify( "std::max_element", resultHost, 1.0 );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost =
               *thrust::max_element( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::max_element", computeThrust );
         verify( "thrust::max_element", resultHost, 1.0 );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorMax( deviceVector );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, 1.0 );

         auto computeCudaET = [ & ]()
         {
            using TNL::max;
            resultDevice = max( deviceView );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, 1.0 );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = *thrust::max_element(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::max_element", computeThrustDevice );
         verify( "thrust::max_element", resultDevice, 1.0 );
#endif
      }
   }

   void
   min()
   {
      benchmark.setOperation( "min" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMin( hostVector );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, 1.0 );

         auto computeET = [ & ]()
         {
            using TNL::min;
            resultHost = min( hostView );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, 1.0 );
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = *std::min_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::min_element", computeSTL );
         verify( "std::min_element", resultHost, 1.0 );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost =
               *thrust::min_element( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::min_element", computeThrust );
         verify( "thrust::min_element", resultHost, 1.0 );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorMin( deviceVector );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, 1.0 );

         auto computeCudaET = [ & ]()
         {
            using TNL::min;
            resultDevice = min( deviceView );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, 1.0 );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = *thrust::min_element(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::min_element", computeThrustDevice );
         verify( "thrust::min_element", resultDevice, 1.0 );
#endif
      }
   }

   void
   absMax()
   {
      benchmark.setOperation( "absMax" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMax( hostVector );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, 1.0 );

         auto computeET = [ & ]()
         {
            using TNL::max;
            resultHost = max( abs( hostView ) );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, 1.0 );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            int index = blasIgamax( size, hostVector.getData(), 1 );
            resultHost = hostVector.getElement( index );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", resultHost, 1.0 );
#endif
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = *std::max_element(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto a, auto b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::max_element", computeSTL );
         verify( "std::max_element", resultHost, 1.0 );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost = *thrust::max_element(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto a, auto b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::max_element", computeThrust );
         verify( "thrust::max_element", resultHost, 1.0 );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorAbsMax( deviceVector );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, 1.0 );

         auto computeCudaET = [ & ]()
         {
            using TNL::max;
            resultDevice = max( abs( deviceView ) );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, 1.0 );

         auto computeCudaCUBLAS = [ & ]()
         {
            int index = 0;
#if defined( __CUDACC__ )
            cublasIgamax( cublasHandle, size, deviceVector.getData(), 1, &index );
#elif defined( __HIP__ )
            hipblasIgamax( hipblasHandle, size, deviceVector.getData(), 1, &index );
#endif
            resultDevice = deviceVector.getElement( index );
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, resultDevice, 1.0 );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = *thrust::max_element(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               [] __cuda_callable__( Real a, Real b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::max_element", computeThrustDevice );
         verify( "thrust::max_element", resultDevice, 1.0 );
#endif
      }
   }

   void
   absMin()
   {
      benchmark.setOperation( "absMin" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMin( hostVector );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, 1.0 );

         auto computeET = [ & ]()
         {
            using TNL::min;
            resultHost = min( abs( hostView ) );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, 1.0 );

#if 0
   #ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            int index = blasIgamin( size, hostVector.getData(), 1 );
            resultHost = hostVector.getElement( index );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", resultHost, 1.0 );
   #endif
#endif
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = *std::min_element(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto a, auto b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::min_element", computeSTL );
         verify( "std::min_element", resultHost, 1.0 );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost = *thrust::min_element(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto a, auto b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::min_element", computeThrust );
         verify( "thrust::min_element", resultHost, 1.0 );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorAbsMin( deviceVector );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, 1.0 );

         auto computeCudaET = [ & ]()
         {
            using TNL::min;
            resultDevice = min( abs( deviceView ) );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, 1.0 );

         auto computeCudaCUBLAS = [ & ]()
         {
            int index = 0;
#if defined( __CUDACC__ )
            cublasIgamin( cublasHandle, size, deviceVector.getData(), 1, &index );
#elif defined( __HIP__ )
            hipblasIgamin( hipblasHandle, size, deviceVector.getData(), 1, &index );
#endif
            resultDevice = deviceVector.getElement( index );
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, resultDevice, 1.0 );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = *thrust::min_element(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               [] __cuda_callable__( Real a, Real b )
               {
                  return std::abs( a ) < std::abs( b );
               } );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::min_element", computeThrustDevice );
         verify( "thrust::min_element", resultDevice, 1.0 );
#endif
      }
   }

   void
   sum()
   {
      benchmark.setOperation( "sum" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorSum( hostVector );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, size );

         auto computeET = [ & ]()
         {
            using TNL::sum;
            resultHost = sum( hostView );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, size );
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = std::reduce( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::reduce", computeSTL );
         verify( "std::reduce", resultHost, size );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost = thrust::reduce( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::reduce", computeThrust );
         verify( "thrust::reduce", resultHost, size );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorSum( deviceVector );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, size );

         auto computeCudaET = [ & ]()
         {
            using TNL::sum;
            resultDevice = sum( deviceView );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, size );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = thrust::reduce(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::reduce", computeThrustDevice );
         verify( "thrust::reduce", resultDevice, size );
#endif
      }
   }

   void
   l1norm()
   {
      benchmark.setOperation( "l1 norm" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 1.0 );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, size );

         auto computeET = [ & ]()
         {
            resultHost = lpNorm( hostView, 1.0 );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, size );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            resultHost = blasGasum( size, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", resultHost, size );
#endif
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = std::transform_reduce(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               0,
               std::plus<>{},
               []( auto v )
               {
                  return std::abs( v );
               } );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::transform_reduce", computeSTL );
         verify( "std::transform_reduce", resultHost, size );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost = thrust::transform_reduce(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto v )
               {
                  return std::abs( v );
               },
               0,
               std::plus<>{} );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::transform_reduce", computeThrust );
         verify( "thrust::transform_reduce", resultHost, size );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorLpNorm( deviceVector, 1.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, size );

         auto computeCudaET = [ & ]()
         {
            resultDevice = lpNorm( deviceView, 1.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, size );

         auto computeCudaCUBLAS = [ & ]()
         {
#if defined( __CUDACC__ )
            cublasGasum( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
#elif defined( __HIP__ )
            hipblasGasum( hipblasHandle, size, deviceVector.getData(), 1, &resultDevice );
#endif
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, resultDevice, size );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = thrust::transform_reduce(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               [] __cuda_callable__( Real v )
               {
                  return std::abs( v );
               },
               0,
               std::plus<>{} );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::transform_reduce", computeThrustDevice );
         verify( "thrust::transform_reduce", resultDevice, size );
#endif
      }
   }

   void
   l2norm()
   {
      benchmark.setOperation( "l2 norm" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 2.0 );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, std::sqrt( size ) );

         auto computeET = [ & ]()
         {
            resultHost = lpNorm( hostView, 2.0 );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, std::sqrt( size ) );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            resultHost = blasGnrm2( size, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", resultHost, std::sqrt( size ) );
#endif
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            const auto sum = std::transform_reduce(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               0,
               std::plus<>{},
               []( auto v )
               {
                  return v * v;
               } );
            resultHost = std::sqrt( sum );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::transform_reduce", computeSTL );
         verify( "std::transform_reduce", resultHost, std::sqrt( size ) );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            const auto sum = thrust::transform_reduce(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto v )
               {
                  return v * v;
               },
               0,
               std::plus<>{} );
            resultHost = std::sqrt( sum );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::transform_reduce", computeThrust );
         verify( "thrust::transform_reduce", resultHost, std::sqrt( size ) );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorLpNorm( deviceVector, 2.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, std::sqrt( size ) );

         auto computeCudaET = [ & ]()
         {
            resultDevice = lpNorm( deviceView, 2.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, std::sqrt( size ) );

         auto computeCudaCUBLAS = [ & ]()
         {
#if defined( __CUDACC__ )
            cublasGnrm2( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
#elif defined( __HIP__ )
            hipblasGnrm2( hipblasHandle, size, deviceVector.getData(), 1, &resultDevice );
#endif
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, resultDevice, std::sqrt( size ) );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            const auto sum = thrust::transform_reduce(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               [] __cuda_callable__( Real v )
               {
                  return v * v;
               },
               0,
               std::plus<>{} );
            resultDevice = std::sqrt( sum );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::transform_reduce", computeThrustDevice );
         verify( "thrust::transform_reduce", resultDevice, std::sqrt( size ) );
#endif
      }
   }

   void
   l3norm()
   {
      benchmark.setOperation( "l3 norm" );
      benchmark.setDatasetSize( datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 3.0 );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, std::cbrt( size ) );

         auto computeET = [ & ]()
         {
            resultHost = lpNorm( hostView, 3.0 );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, std::cbrt( size ) );
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            const auto sum = std::transform_reduce(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               0,
               std::plus<>{},
               []( auto v )
               {
                  return std::pow( v, 3 );
               } );
            resultHost = std::cbrt( sum );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::transform_reduce", computeSTL );
         verify( "std::transform_reduce", resultHost, std::cbrt( size ) );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            const auto sum = thrust::transform_reduce(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               []( auto v )
               {
                  return std::pow( v, 3 );
               },
               0,
               std::plus<>{} );
            resultHost = std::cbrt( sum );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::transform_reduce", computeThrust );
         verify( "thrust::transform_reduce", resultHost, std::cbrt( size ) );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getVectorLpNorm( deviceVector, 3.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, std::cbrt( size ) );

         auto computeCudaET = [ & ]()
         {
            resultDevice = lpNorm( deviceView, 3.0 );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, std::cbrt( size ) );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            const auto sum = thrust::transform_reduce(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               [] __cuda_callable__( Real v )
               {
                  return std::pow( v, 3 );
               },
               0,
               std::plus<>{} );
            resultDevice = std::cbrt( sum );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::transform_reduce", computeThrustDevice );
         verify( "thrust::transform_reduce", resultDevice, std::cbrt( size ) );
#endif
      }
   }

   void
   scalarProduct()
   {
      benchmark.setOperation( "scalar product" );
      benchmark.setDatasetSize( 2 * datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getScalarProduct( hostVector, hostVector2 );
         };
         benchmark.time< Devices::Host >( reset1, "legacy", computeLegacy );
         verify( "legacy", resultHost, size );

         auto computeET = [ & ]()
         {
            resultHost = ( hostVector, hostVector2 );
         };
         benchmark.time< Devices::Host >( reset1, "ET", computeET );
         verify( "ET", resultHost, size );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            resultHost = blasGdot( size, hostVector.getData(), 1, hostVector2.getData(), 1 );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", resultHost, size );
#endif
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         auto computeSTL = [ & ]()
         {
            resultHost = std::transform_reduce(
               STDEXEC hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               hostVector2.getData(),
               0,
               std::plus<>{},
               std::multiplies<>{} );
         };
         benchmark.time< StdAlgorithmsDevice >( reset1, "std::transform_reduce", computeSTL );
         verify( "std::transform_reduce", resultHost, size );
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         auto computeThrust = [ & ]()
         {
            resultHost = thrust::inner_product(
               thrust::host,
               hostVector.getData(),
               hostVector.getData() + hostVector.getSize(),
               hostVector2.getData(),
               0,
               std::plus<>{},
               std::multiplies<>{} );
         };
         benchmark.time< ThrustHostDevice >( reset1, "thrust::inner_product", computeThrust );
         verify( "thrust::inner_product", resultHost, size );
      }
#endif
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            resultDevice = Benchmarks::CommonVectorOperations< Devices::GPU >::getScalarProduct( deviceVector, deviceVector2 );
         };
         benchmark.time< Devices::GPU >( reset1, "legacy", computeCudaLegacy );
         verify( "legacy", resultDevice, size );

         auto computeCudaET = [ & ]()
         {
            resultDevice = ( deviceView, deviceView2 );
         };
         benchmark.time< Devices::GPU >( reset1, "ET", computeCudaET );
         verify( "ET", resultDevice, size );

         auto computeCudaCUBLAS = [ & ]()
         {
#if defined( __CUDACC__ )
            cublasGdot( cublasHandle, size, deviceVector.getData(), 1, deviceVector2.getData(), 1, &resultDevice );
#elif defined( __HIP__ )
            hipblasGdot( hipblasHandle, size, deviceVector.getData(), 1, deviceVector2.getData(), 1, &resultDevice );
#endif
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, resultDevice, size );

#ifdef HAVE_THRUST
         auto computeThrustDevice = [ & ]()
         {
            resultDevice = thrust::inner_product(
               thrust::device,
               thrust::device_pointer_cast( deviceVector.getData() ),
               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
               thrust::device_pointer_cast( deviceVector2.getData() ),
               0,
               std::plus<>{},
               std::multiplies<>{} );
         };
         benchmark.time< Devices::GPU >( reset1, "thrust::inner_product", computeThrustDevice );
         verify( "thrust::inner_product", resultDevice, size );
#endif
      }
   }

   void
   scalarMultiplication()
   {
      benchmark.setOperation( "scalar multiplication" );
      benchmark.setDatasetSize( 2 * datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeTNL = [ & ]()
         {
            const Real alpha = 0.5;
            hostVector *= alpha;
         };
         benchmark.time< Devices::Host >( reset1, "TNL", computeTNL );
         verify( "TNL", hostVector, 0.5 );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            const Real alpha = 0.5;
            blasGscal( hostVector.getSize(), alpha, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( reset1, "BLAS", computeBLAS );
         verify( "BLAS", hostVector, 0.5 );
#endif
      }
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaTNL = [ & ]()
         {
            deviceVector *= 0.5;
         };
         benchmark.time< Devices::GPU >( reset1, "TNL", computeCudaTNL );
         verify( "TNL", deviceVector, 0.5 );

         auto computeCudaCUBLAS = [ & ]()
         {
            const Real alpha = 0.5;
#if defined( __CUDACC__ )
            cublasGscal( cublasHandle, size, &alpha, deviceVector.getData(), 1 );
#elif defined( __HIP__ )
            hipblasGscal( hipblasHandle, size, &alpha, deviceVector.getData(), 1 );
#endif
         };
         benchmark.time< Devices::GPU >( reset1, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, deviceVector, 0.5 );
      }
   }

   void
   vectorAddition()
   {
      benchmark.setOperation( "vector addition" );
      benchmark.setDatasetSize( 3 * datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::Host >( resetAll, "legacy", computeLegacy );
         verify( "legacy", hostVector, 2.0 );

         auto computeET = [ & ]()
         {
            hostView += hostView2;
         };
         benchmark.time< Devices::Host >( resetAll, "ET", computeET );
         verify( "ET", hostVector, 2.0 );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            const Real alpha = 1.0;
            blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( resetAll, "BLAS", computeBLAS );
         verify( "BLAS", hostVector, 2.0 );
#endif
      }
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::GPU >( resetAll, "legacy", computeCudaLegacy );
         verify( "legacy", deviceVector, 2.0 );

         auto computeCudaET = [ & ]()
         {
            deviceView += deviceView2;
         };
         benchmark.time< Devices::GPU >( resetAll, "ET", computeCudaET );
         verify( "ET", deviceVector, 2.0 );

         auto computeCudaCUBLAS = [ & ]()
         {
            const Real alpha = 1.0;
#if defined( __CUDACC__ )
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
#elif defined( __HIP__ )
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
#endif
         };
         benchmark.time< Devices::GPU >( resetAll, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, deviceVector, 2.0 );
      }
   }

   void
   twoVectorsAddition()
   {
      benchmark.setOperation( "two vectors addition" );
      benchmark.setDatasetSize( 4 * datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector3, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::Host >( resetAll, "legacy", computeLegacy );
         verify( "legacy", hostVector, 3.0 );

         auto computeParallelFor = [ & ]()
         {
            Algorithms::parallelFor< Devices::Host >(
               0,
               hostVector.getSize(),
               [ = ]( const Index i )
               {
                  hostView[ i ] += hostView2[ i ] + hostView3[ i ];
               } );
         };
         benchmark.time< Devices::Host >( resetAll, "parallel for", computeParallelFor );
         verify( "parallel for", hostVector, 3.0 );

         auto computeET = [ & ]()
         {
            hostView += hostView2 + hostView3;
         };
         benchmark.time< Devices::Host >( resetAll, "ET", computeET );
         verify( "ET", hostVector, 3.0 );

         auto computeLinearCombination = [ & ]()
         {
            hostView += Containers::linearCombination< TwoVectorsCoefficients< Real > >( hostView2, hostView3 );
         };
         benchmark.time< Devices::Host >( resetAll, "linear combination", computeLinearCombination );
         verify( "linear combination", hostVector, 3.0 );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            const Real alpha = 1.0;
            blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
            blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( resetAll, "BLAS", computeBLAS );
         verify( "BLAS", hostVector, 3.0 );
#endif
      }
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector3, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::GPU >( resetAll, "legacy", computeCudaLegacy );
         verify( "legacy", deviceVector, 3.0 );

         // we cannot capture *this (it leads to segfault) in a lambda, so we need to capture the device vector by value
         auto d1 = deviceVector.getView();
         auto d2 = deviceVector2.getView();
         auto d3 = deviceVector3.getView();
         auto computeCudaParallelFor = [ & ]()
         {
            Algorithms::parallelFor< Devices::GPU >(
               0,
               deviceView.getSize(),
               [ = ] __cuda_callable__( const Index i ) mutable
               {
                  d1[ i ] += d2[ i ] + d3[ i ];
               } );
         };
         benchmark.time< Devices::GPU >( resetAll, "parallel for", computeCudaParallelFor );
         verify( "parallel for", deviceVector, 3.0 );

         auto computeCudaET = [ & ]()
         {
            deviceView += deviceView2 + deviceView3;
         };
         benchmark.time< Devices::GPU >( resetAll, "ET", computeCudaET );
         verify( "ET", deviceVector, 3.0 );

         auto computeCudaLinearCombination = [ & ]()
         {
            deviceView += Containers::linearCombination< TwoVectorsCoefficients< Real > >( deviceView2, deviceView3 );
         };
         benchmark.time< Devices::GPU >( resetAll, "linear combination", computeCudaLinearCombination );
         verify( "linear combination", deviceVector, 3.0 );

         auto computeCudaCUBLAS = [ & ]()
         {
            const Real alpha = 1.0;
#if defined( __CUDACC__ )
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
#elif defined( __HIP__ )
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
#endif
         };
         benchmark.time< Devices::GPU >( resetAll, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, deviceVector, 3.0 );
      }
   }

   void
   threeVectorsAddition()
   {
      benchmark.setOperation( "three vectors addition" );
      benchmark.setDatasetSize( 5 * datasetSize );

      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         auto computeLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector3, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::Host >::addVector(
               hostVector, hostVector4, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::Host >( resetAll, "legacy", computeLegacy );
         verify( "legacy", hostVector, 4.0 );

         auto computeParallelFor = [ & ]()
         {
            Algorithms::parallelFor< Devices::Host >(
               0,
               hostVector.getSize(),
               [ = ] __cuda_callable__( const Index i )
               {
                  hostView[ i ] += hostView2[ i ] + hostView3[ i ] + hostView4[ i ];
               } );
         };
         benchmark.time< Devices::Host >( resetAll, "parallel for", computeParallelFor );
         verify( "parallel for", hostVector, 4.0 );

         auto computeET = [ & ]()
         {
            hostView += hostView2 + hostView3 + hostView4;
         };
         benchmark.time< Devices::Host >( resetAll, "ET", computeET );
         verify( "ET", hostVector, 4.0 );

         auto computeLinearCombination = [ & ]()
         {
            hostView += Containers::linearCombination< ThreeVectorsCoefficients< Real > >( hostView2, hostView3, hostView4 );
         };
         benchmark.time< Devices::Host >( resetAll, "linear combination", computeLinearCombination );
         verify( "linear combination", hostVector, 4.0 );

#ifdef HAVE_BLAS
         auto computeBLAS = [ & ]()
         {
            const Real alpha = 1.0;
            blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
            blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
            blasGaxpy( size, alpha, hostVector4.getData(), 1, hostVector.getData(), 1 );
         };
         benchmark.time< Devices::Host >( resetAll, "BLAS", computeBLAS );
         verify( "BLAS", hostVector, 4.0 );
#endif
      }
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         auto computeCudaLegacy = [ & ]()
         {
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector2, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector3, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
            Benchmarks::VectorOperations< Devices::GPU >::addVector(
               deviceVector, deviceVector4, static_cast< Real >( 1 ), static_cast< Real >( 1 ) );
         };
         benchmark.time< Devices::GPU >( resetAll, "legacy", computeCudaLegacy );
         verify( "legacy", deviceVector, 4.0 );

         // we cannot capture *this (it leads to segfault) in a lambda, so we need to capture the device vector by value
         auto d1 = deviceVector.getView();
         auto d2 = deviceVector2.getView();
         auto d3 = deviceVector3.getView();
         auto d4 = deviceVector4.getView();
         auto computeCudaParallelFor = [ & ]()
         {
            Algorithms::parallelFor< Devices::GPU >(
               0,
               deviceVector.getSize(),
               [ = ] __cuda_callable__( const Index i ) mutable
               {
                  d1[ i ] += d2[ i ] + d3[ i ] + d4[ i ];
               } );
         };
         benchmark.time< Devices::GPU >( resetAll, "parallel for", computeCudaParallelFor );
         verify( "parallel for", deviceVector, 4.0 );

         auto computeCudaET = [ & ]()
         {
            deviceView += deviceView2 + deviceView3 + deviceView4;
         };
         benchmark.time< Devices::GPU >( resetAll, "ET", computeCudaET );
         verify( "ET", deviceVector, 4.0 );

         auto computeCudaLinearCombination = [ & ]()
         {
            deviceView +=
               Containers::linearCombination< ThreeVectorsCoefficients< Real > >( deviceView2, deviceView3, deviceView4 );
         };
         benchmark.time< Devices::GPU >( resetAll, "linear combination", computeCudaLinearCombination );
         verify( "linear combination", deviceVector, 4.0 );

         auto computeCudaCUBLAS = [ & ]()
         {
            const Real alpha = 1.0;
#if defined( __CUDACC__ )
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
            cublasGaxpy( cublasHandle, size, &alpha, deviceVector4.getData(), 1, deviceVector.getData(), 1 );
#elif defined( __HIP__ )
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
            hipblasGaxpy( hipblasHandle, size, &alpha, deviceVector4.getData(), 1, deviceVector.getData(), 1 );
#endif
         };
         benchmark.time< Devices::GPU >( resetAll, gpuBlasName, computeCudaCUBLAS );
         verify( gpuBlasName, deviceVector, 4.0 );
      }
   }
};

template< typename Device, typename Real = double, typename Index = int >
void
benchmarkVectorOperations( Benchmark& benchmark, const long& size )
{
   VectorOperationsBenchmark< Device, Real, Index > ops( benchmark, size );
   ops.max();
   ops.min();
   ops.absMax();
   ops.absMin();
   ops.sum();
   ops.l1norm();
   ops.l2norm();
   ops.l3norm();
   ops.scalarProduct();
   ops.scalarMultiplication();
   ops.vectorAddition();
   ops.twoVectorsAddition();
   ops.threeVectorsAddition();
}

}  // namespace TNL::Benchmarks
