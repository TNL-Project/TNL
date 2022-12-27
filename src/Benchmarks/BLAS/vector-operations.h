// Implemented by: Jakub Klinkovsky

#pragma once

#include <cstdlib>     // srand48
#include <algorithm>   // std::max_element, std::min_element, std::transform, etc.
#include <numeric>     // std::reduce, std::transform_reduce, std::partial_sum, std::inclusive_scan, std::exclusive_scan
#include <execution>   // std::execution policies
#include <functional>  // std::function

#if defined( HAVE_TBB ) && defined( __cpp_lib_parallel_algorithm )
   #define STDEXEC std::execution::par_unseq,
#else
   #define STDEXEC
#endif

#include <TNL/Benchmarks/Benchmarks.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/scan.h>
#include "CommonVectorOperations.h"
#include "VectorOperations.h"

#ifdef HAVE_BLAS
   #include "blasWrappers.h"
#endif

#ifdef __CUDACC__
   #include "cublasWrappers.h"
#endif

namespace TNL {
namespace Benchmarks {

template< typename Real = double, typename Index = int >
class VectorOperationsBenchmark
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, Index >;
   using SequentialView = Containers::VectorView< Real, Devices::Sequential, Index >;
   using HostView = Containers::VectorView< Real, Devices::Host, Index >;
   using CudaView = Containers::VectorView< Real, Devices::Cuda, Index >;

   Benchmark<>& benchmark;
   long size = 0;
   double datasetSize = 0;

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

#ifdef __CUDACC__
   cublasHandle_t cublasHandle;
#endif

public:
   VectorOperationsBenchmark( Benchmark<>& benchmark, const long& size )
   : benchmark( benchmark ), size( size ), datasetSize( (double) size * sizeof( Real ) / oneGB )
   {
      hostVector.setSize( size );
      hostVector2.setSize( size );
      hostVector3.setSize( size );
      hostVector4.setSize( size );
#ifdef __CUDACC__
      deviceVector.setSize( size );
      deviceVector2.setSize( size );
      deviceVector3.setSize( size );
      deviceVector4.setSize( size );
#endif

      hostView.bind( hostVector );
      hostView2.bind( hostVector2 );
      hostView3.bind( hostVector3 );
      hostView4.bind( hostVector4 );
#ifdef __CUDACC__
      deviceView.bind( deviceVector );
      deviceView2.bind( deviceVector2 );
      deviceView3.bind( deviceVector3 );
      deviceView4.bind( deviceVector4 );
#endif

      // reset functions
      // (Make sure to always use some in benchmarks, even if it's not necessary
      // to assure correct result - it helps to clear cache and avoid optimizations
      // of the benchmark loop.)
      reset1 = [ & ]()
      {
         hostVector.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector.setValue( 1.0 );
#endif
         // A relatively harmless call to keep the compiler from realizing we
         // don't actually do any useful work with the result of the reduction.
         srand( static_cast< unsigned int >( resultHost ) );
         resultHost = resultDevice = 0.0;
      };
      reset2 = [ & ]()
      {
         hostVector2.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector2.setValue( 1.0 );
#endif
      };
      reset3 = [ & ]()
      {
         hostVector3.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector3.setValue( 1.0 );
#endif
      };
      reset4 = [ & ]()
      {
         hostVector4.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector4.setValue( 1.0 );
#endif
      };

      resetAll = [ & ]()
      {
         reset1();
         reset2();
         reset3();
         reset4();
      };

      resetAll();

#ifdef __CUDACC__
      cublasCreate( &cublasHandle );
#endif
   }

   ~VectorOperationsBenchmark()
   {
#ifdef __CUDACC__
      cublasDestroy( cublasHandle );
#endif
   }

   void
   max()
   {
      benchmark.setOperation( "max", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMax( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         using TNL::max;
         resultHost = max( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

      auto computeSTL = [ & ]()
      {
         resultHost = *std::max_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::max_element", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMax( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         using TNL::max;
         resultDevice = max( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
#endif
   }

   void
   min()
   {
      benchmark.setOperation( "min", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMin( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         using TNL::min;
         resultHost = min( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, 1.0 );

      auto computeSTL = [ & ]()
      {
         resultHost = *std::min_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::min_element", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMin( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         using TNL::min;
         resultDevice = min( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
#endif
   }

   void
   absMax()
   {
      benchmark.setOperation( "absMax", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMax( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         using TNL::max;
         resultHost = max( abs( hostView ) );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         int index = blasIgamax( size, hostVector.getData(), 1 );
         resultHost = hostVector.getElement( index );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = *std::max_element( STDEXEC hostVector.getData(),
                                         hostVector.getData() + hostVector.getSize(),
                                         []( auto a, auto b )
                                         {
                                            return std::abs( a ) < std::abs( b );
                                         } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::max_element", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMax( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         using TNL::max;
         resultDevice = max( abs( deviceView ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         int index = 0;
         cublasIgamax( cublasHandle, size, deviceVector.getData(), 1, &index );
         resultDevice = deviceVector.getElement( index );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   absMin()
   {
      benchmark.setOperation( "absMin", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMin( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         using TNL::min;
         resultHost = min( abs( hostView ) );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#if 0
   #ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         int index = blasIgamin( size, hostVector.getData(), 1 );
         resultHost = hostVector.getElement( index );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
   #endif
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = *std::min_element( STDEXEC hostVector.getData(),
                                         hostVector.getData() + hostVector.getSize(),
                                         []( auto a, auto b )
                                         {
                                            return std::abs( a ) < std::abs( b );
                                         } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::min_element", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMin( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         using TNL::min;
         resultDevice = min( abs( deviceView ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         int index = 0;
         cublasIgamin( cublasHandle, size, deviceVector.getData(), 1, &index );
         resultDevice = deviceVector.getElement( index );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   sum()
   {
      benchmark.setOperation( "sum", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorSum( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         using TNL::sum;
         resultHost = sum( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

      auto computeSTL = [ & ]()
      {
         resultHost = std::reduce( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::reduce", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorSum( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto copmuteCudaET = [ & ]()
      {
         using TNL::sum;
         resultDevice = sum( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", copmuteCudaET );
#endif
   }

   void
   l1norm()
   {
      benchmark.setOperation( "l1 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 1.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 1.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGasum( size, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = std::transform_reduce( STDEXEC hostVector.getData(),
                                             hostVector.getData() + hostVector.getSize(),
                                             0,
                                             std::plus<>{},
                                             []( auto v )
                                             {
                                                return std::abs( v );
                                             } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 1.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 1.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGasum( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   l2norm()
   {
      benchmark.setOperation( "l2 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 2.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 2.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGnrm2( size, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
#endif

      auto computeSTL = [ & ]()
      {
         const auto sum = std::transform_reduce( STDEXEC hostVector.getData(),
                                                 hostVector.getData() + hostVector.getSize(),
                                                 0,
                                                 std::plus<>{},
                                                 []( auto v )
                                                 {
                                                    return v * v;
                                                 } );
         resultHost = std::sqrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 2.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 2.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGnrm2( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   l3norm()
   {
      benchmark.setOperation( "l3 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 3.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 3.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

      auto computeSTL = [ & ]()
      {
         const auto sum = std::transform_reduce( STDEXEC hostVector.getData(),
                                                 hostVector.getData() + hostVector.getSize(),
                                                 0,
                                                 std::plus<>{},
                                                 []( auto v )
                                                 {
                                                    return v * v * v;
                                                 } );
         resultHost = std::cbrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 3.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 3.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
#endif
   }

   void
   scalarProduct()
   {
      benchmark.setOperation( "scalar product", 2 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getScalarProduct( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         resultHost = ( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGdot( size, hostVector.getData(), 1, hostVector2.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = std::transform_reduce( STDEXEC hostVector.getData(),
                                             hostVector.getData() + hostVector.getSize(),
                                             hostVector2.getData(),
                                             0,
                                             std::plus<>{},
                                             std::multiplies<>{} );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getScalarProduct( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         resultDevice = ( deviceView, deviceView2 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGdot( cublasHandle, size, deviceVector.getData(), 1, deviceVector2.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   scalarMultiplication()
   {
      benchmark.setOperation( "scalar multiplication", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         hostVector *= 0.5;
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         blasGscal( hostVector.getSize(), (Real) 0.5, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
#endif

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         deviceVector *= 0.5;
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 0.5;
         cublasGscal( cublasHandle, size, &alpha, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   vectorAddition()
   {
      benchmark.setOperation( "vector addition", 3 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         hostView += hostView2;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   twoVectorsAddition()
   {
      benchmark.setOperation( "two vectors addition", 4 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         hostView += hostView2 + hostView3;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2 + deviceView3;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   threeVectorsAddition()
   {
      benchmark.setOperation( "three vectors addition", 5 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector4, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );

      auto computeET = [ & ]()
      {
         hostView += hostView2 + hostView3 + hostView4;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector4.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector4, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2 + deviceView3 + deviceView4;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector4.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
#endif
   }

   void
   inclusiveScanInplace()
   {
      benchmark.setOperation( "inclusive scan (inplace)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inplaceInclusiveScan( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

      auto computeSequential = [ & ]()
      {
         SequentialView view;
         view.bind( hostVector.getData(), hostVector.getSize() );
         Algorithms::inplaceInclusiveScan( view );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU sequential", computeSequential );

      auto computeSTL_partial_sum = [ & ]()
      {
         std::partial_sum( hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::partial_sum", computeSTL_partial_sum );

      auto computeSTL = [ & ]()
      {
         std::inclusive_scan( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::inclusive_scan", computeSTL );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inplaceInclusiveScan( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
#endif
   }

   void
   inclusiveScanOneVector()
   {
      benchmark.setOperation( "inclusive scan (1 vector)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

      auto computeSTL_partial_sum = [ & ]()
      {
         std::partial_sum( hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( resetAll, "CPU std::partial_sum", computeSTL_partial_sum );

      auto computeSTL = [ & ]()
      {
         std::inclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( resetAll, "CPU std::inclusive_scan", computeSTL );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }

   void
   inclusiveScanTwoVectors()
   {
      benchmark.setOperation( "inclusive scan (2 vectors)", 3 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector + hostVector2, hostVector3 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector + deviceVector2, deviceVector3 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }

   void
   inclusiveScanThreeVectors()
   {
      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
      };
      benchmark.setOperation( "inclusive scan (3 vectors)", 4 * datasetSize );
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }

   void
   exclusiveScanInplace()
   {
      benchmark.setOperation( "exclusive scan (inplace)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inplaceExclusiveScan( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );

      auto computeSequential = [ & ]()
      {
         SequentialView view;
         view.bind( hostVector.getData(), hostVector.getSize() );
         Algorithms::inplaceExclusiveScan( view );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU sequential", computeSequential );

      auto computeSTL = [ & ]()
      {
         std::exclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData(), 0 );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::exclusive_scan", computeSTL );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inplaceExclusiveScan( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
#endif
   }

   void
   exclusiveScanOneVector()
   {
      benchmark.setOperation( "exclusive scan (1 vector)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

      auto computeSTL = [ & ]()
      {
         std::exclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData(), 0 );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::exclusive_scan", computeSTL );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }

   void
   exclusiveScanTwoVectors()
   {
      benchmark.setOperation( "exclusive scan (2 vectors)", 3 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector + hostVector2, hostVector3 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector + deviceVector2, deviceVector3 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }

   void
   exclusiveScanThreeVectors()
   {
      benchmark.setOperation( "exclusive scan (3 vectors)", 4 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
#endif
   }
};

template< typename Real = double, typename Index = int >
void
benchmarkVectorOperations( Benchmark<>& benchmark, const long& size )
{
   VectorOperationsBenchmark< Real, Index > ops( benchmark, size );
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
   ops.inclusiveScanInplace();
   ops.inclusiveScanOneVector();
   ops.inclusiveScanTwoVectors();
   ops.inclusiveScanThreeVectors();
   ops.exclusiveScanInplace();
   ops.exclusiveScanOneVector();
   ops.exclusiveScanTwoVectors();
   ops.exclusiveScanThreeVectors();
}

}  // namespace Benchmarks
}  // namespace TNL
