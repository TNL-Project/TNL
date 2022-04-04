
#pragma once

#ifdef HAVE_CUDA

   #include <TNL/Devices/Cuda.h>
   #include <TNL/Containers/StaticVector.h>
   #include <TNL/Cuda/LaunchHelpers.h>

/**
 * There are several pitfalls with such configuration.
 *
 * 1. At first we don't use shared memory
 * 2. At second we don't control block size, so we may launch extremely small kernels or otherwise we can launch extremely large
 * kernels.
 */

template< int Dimension, typename Device >
struct Convolution;

template< typename Index,
          typename Real,
          typename FetchData,
          typename FetchBoundary,
          typename FetchKernel,
          typename Convolve,
          typename Store >
__global__
static void
convolution1D( Index kernelWidth,
               Index endX,
               FetchData fetchData,
               FetchBoundary fetchBoundary,
               FetchKernel fetchKernel,
               Convolve convolve,
               Store store )
{
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   if( ix >= endX )
      return;

   Index radius = kernelWidth >> 1;

   Real result = 0;

   for( Index i = -radius; i <= radius; i++ ) {
      Index elementIndex = i + ix;
      Index kernelIndex = i + radius;

      if( elementIndex < 0 || elementIndex >= endX ) {
         result = convolve( result, fetchBoundary( elementIndex ), fetchKernel( kernelIndex ) );
      }
      else {
         result = convolve( result, fetchData( elementIndex ), fetchKernel( kernelIndex ) );
      }
   }

   store( ix, result );
}

template< typename Index,
          typename Real,
          typename FetchData,
          typename FetchBoundary,
          typename FetchKernel,
          typename Convolve,
          typename Store >
__global__
static void
convolution2D( Index kernelWidth,
               Index kernelHeight,
               Index endX,
               Index endY,
               FetchData fetchData,
               FetchBoundary fetchBoundary,
               FetchKernel fetchKernel,
               Convolve convolve,
               Store store )
{
   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   if( ix >= endX || iy >= endY )
      return;

   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Real result = 0;

   for( Index j = -radiusY; j <= radiusY; j++ ) {
      Index elementIndexY = j + iy;
      Index kernelIndexY = j + radiusY;

      for( Index i = -radiusX; i <= radiusX; i++ ) {
         Index elementIndexX = i + ix;
         Index kernelIndexX = i + radiusX;

         if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY ) {
            result =
               convolve( result, fetchBoundary( elementIndexX, elementIndexY ), fetchKernel( kernelIndexX, kernelIndexY ) );
         }
         else {
            result = convolve( result, fetchData( elementIndexX, elementIndexY ), fetchKernel( kernelIndexX, kernelIndexY ) );
         }
      }
   }

   store( ix, iy, result );
}

template< typename Index,
          typename Real,
          typename FetchData,
          typename FetchBoundary,
          typename FetchKernel,
          typename Convolve,
          typename Store >
__global__
static void
convolution3D( Index kernelWidth,
               Index kernelHeight,
               Index kernelDepth,
               Index endX,
               Index endY,
               Index endZ,
               FetchData fetchData,
               FetchBoundary fetchBoundary,
               FetchKernel fetchKernel,
               Convolve convolve,
               Store store )
{
   Index iz = threadIdx.z + blockIdx.z * blockDim.z;
   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   if( ix >= endX || iy >= endY || iz >= endZ )
      return;

   Index radiusZ = kernelDepth >> 1;
   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Real result = 0;

   for( Index k = -radiusZ; k <= radiusZ; k++ ) {
      Index elementIndexZ = k + iz;
      Index kernelIndexZ = k + radiusZ;

      for( Index j = -radiusY; j <= radiusY; j++ ) {
         Index elementIndexY = j + iy;
         Index kernelIndexY = j + radiusY;

         for( Index i = -radiusX; i <= radiusX; i++ ) {
            Index elementIndexX = i + ix;
            Index kernelIndexX = i + radiusX;

            if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY || elementIndexZ < 0
                || elementIndexZ >= endZ )
            {
               result = convolve( result,
                                  fetchBoundary( elementIndexX, elementIndexY, elementIndexZ ),
                                  fetchKernel( kernelIndexX, kernelIndexY, kernelIndexZ ) );
            }
            else {
               result = convolve( result,
                                  fetchData( elementIndexX, elementIndexY, elementIndexZ ),
                                  fetchKernel( kernelIndexX, kernelIndexY, kernelIndexZ ) );
            }
         }
      }
   }

   store( ix, iy, iz, result );
}

template< int Dimension, typename Device >
struct Convolution;

template<>
struct Convolution< 1, TNL::Devices::Cuda >
{
public:
   template< typename Index >
   using Vector = TNL::Containers::StaticVector< 1, Index >;

   template< typename Index, typename Real >
   static void
   setup( TNL::Cuda::LaunchConfiguration& configuration, const Vector< Index >& dimensions, const Vector< Index >& kernelSize )
   {
      configuration.dynamicSharedMemorySize = 0;

      // TODO: - Benchmark the best value
      configuration.blockSize.x = kernelSize.x();
      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
   }

   template< typename Index,
             typename Real,
             typename FetchData,
             typename FetchBoundary,
             typename FetchKernel,
             typename Convolve,
             typename Store >
   static void
   execute( const Vector< Index >& dimensions,
            const Vector< Index >& kernelSize,
            FetchData&& fetchData,
            FetchBoundary&& fetchBoundary,
            FetchKernel&& fetchKernel,
            Convolve&& convolve,
            Store&& store )
   {
      TNL::Cuda::LaunchConfiguration configuration;

      setup< Index, Real >( configuration, dimensions, kernelSize );

      constexpr auto kernel = convolution1D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >(
         kernel, 0, configuration, kernelSize.x(), dimensions.x(), fetchData, fetchBoundary, fetchKernel, convolve, store );
   };
};

template<>
struct Convolution< 2, TNL::Devices::Cuda >
{
public:
   template< typename Index >
   using Vector = TNL::Containers::StaticVector< 2, Index >;

   template< typename Index, typename Real >
   static void
   setup( TNL::Cuda::LaunchConfiguration& configuration, const Vector< Index >& dimensions, const Vector< Index >& kernelSize )
   {
      configuration.dynamicSharedMemorySize = 0;

      configuration.blockSize.x = kernelSize.x();
      configuration.blockSize.y = kernelSize.y();

      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
      configuration.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.y(), configuration.blockSize.y ) );
   }

   template< typename Index,
             typename Real,
             typename FetchData,
             typename FetchBoundary,
             typename FetchKernel,
             typename Convolve,
             typename Store >
   static void
   execute( const Vector< Index >& dimensions,
            const Vector< Index >& kernelSize,
            FetchData&& fetchData,
            FetchBoundary&& fetchBoundary,
            FetchKernel&& fetchKernel,
            Convolve&& convolve,
            Store&& store )
   {
      TNL::Cuda::LaunchConfiguration configuration;

      setup< Index, Real >( configuration, dimensions, kernelSize );

      constexpr auto kernel = convolution2D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       configuration,
                                       kernelSize.x(),
                                       kernelSize.y(),
                                       dimensions.x(),
                                       dimensions.y(),
                                       fetchData,
                                       fetchBoundary,
                                       fetchKernel,
                                       convolve,
                                       store );
   };
};

template<>
struct Convolution< 3, TNL::Devices::Cuda >
{
public:
   template< typename Index >
   using Vector = TNL::Containers::StaticVector< 3, Index >;

   template< typename Index, typename Real >
   static void
   setup( TNL::Cuda::LaunchConfiguration& configuration, const Vector< Index >& dimensions, const Vector< Index >& kernelSize )
   {
      configuration.dynamicSharedMemorySize = 0;

      // TODO: - Benchmark the best value
      configuration.blockSize.x = kernelSize.x();
      configuration.blockSize.y = kernelSize.y();
      configuration.blockSize.z = kernelSize.z();

      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
      configuration.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.y(), configuration.blockSize.y ) );
      configuration.gridSize.z =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.z(), configuration.blockSize.z ) );
   }

   template< typename Index,
             typename Real,
             typename FetchData,
             typename FetchBoundary,
             typename FetchKernel,
             typename Convolve,
             typename Store >
   static void
   execute( const Vector< Index >& dimensions,
            const Vector< Index >& kernelSize,
            FetchData&& fetchData,
            FetchBoundary&& fetchBoundary,
            FetchKernel&& fetchKernel,
            Convolve&& convolve,
            Store&& store )
   {
      TNL::Cuda::LaunchConfiguration configuration;

      setup< Index, Real >( configuration, dimensions, kernelSize );

      constexpr auto kernel = convolution3D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       configuration,
                                       kernelSize.x(),
                                       kernelSize.y(),
                                       kernelSize.z(),
                                       dimensions.x(),
                                       dimensions.y(),
                                       dimensions.z(),
                                       fetchData,
                                       fetchBoundary,
                                       fetchKernel,
                                       convolve,
                                       store );
   };
};

#endif
