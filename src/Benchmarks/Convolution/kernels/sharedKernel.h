
#pragma once

#ifdef HAVE_CUDA

#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/SharedMemory.h>

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
      Index kernelElementCount = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ )
         kernelElementCount *= kernelSize[ i ];

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

      configuration.blockSize.x = kernelSize.x();
      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
   }
};

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
   Real* shared = TNL::Cuda::getSharedMemory< Real >();

   Index radius = kernelWidth >> 1;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   // The size of the block is equal to the kernel size
   shared[ threadIdx.x ] = fetchKernel( threadIdx.x );

   __syncthreads();

   Real result = 0;

   for( Index i = -radius; i <= radius; i++ ) {
      Index elementIndex = i + ix;
      Index kernelIndex = i + radius;

      if( elementIndex < 0 || elementIndex >= endX ) {
         result = convolve( result, fetchBoundary( elementIndex ), shared[ kernelIndex ] );
      }
      else {
         result = convolve( result, fetchData( elementIndex ), shared[ kernelIndex ] );
      }
   }

   store( ix, result );
}

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
      Index kernelElementCount = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ )
         kernelElementCount *= kernelSize[ i ];

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

      configuration.blockSize.x = kernelSize.x();
      configuration.blockSize.y = kernelSize.y();

      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
      configuration.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.y(), configuration.blockSize.y ) );
   }
};

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
   Real* shared = TNL::Cuda::getSharedMemory< Real >();

   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   Index threadIndex = threadIdx.x + blockDim.x * threadIdx.y;

   // The size of the block is equal to the kernel size
   shared[ threadIndex ] = fetchKernel( threadIdx.x, threadIdx.y );

   __syncthreads();

   Real result = 0;

   for( Index j = -radiusY; j <= radiusY; j++ ) {
      Index elementIndexY = j + iy;
      Index kernelIndexY = j + radiusY;

      for( Index i = -radiusX; i <= radiusX; i++ ) {
         Index elementIndexX = i + ix;
         Index kernelIndexX = i + radiusX;

         Index threadIndex = kernelIndexX + kernelWidth * kernelIndexY;

         if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY ) {
            result = convolve( result, fetchBoundary( elementIndexX, elementIndexY ), shared[ threadIndex ] );
         }
         else {
            result = convolve( result, fetchData( elementIndexX, elementIndexY ), shared[ threadIndex ] );
         }
      }
   }

   store( ix, iy, result );
}

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
      Index kernelElementCount = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ )
         kernelElementCount *= kernelSize[ i ];

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

      configuration.blockSize.x = kernelSize.x();
      configuration.blockSize.y = kernelSize.y();
      configuration.blockSize.z = kernelSize.z();

      configuration.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), configuration.blockSize.x ) );
      configuration.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.y(), configuration.blockSize.y ) );
      configuration.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.z(), configuration.blockSize.z ) );
   }
};

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
   Real* shared = TNL::Cuda::getSharedMemory< Real >();

   Index iz = threadIdx.z + blockIdx.z * blockDim.z;
   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   Index radiusZ = kernelDepth >> 1;
   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index threadIndex = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

   printf( "%d\n", threadIndex );

   // The size of the block is equal to the kernel size
   shared[ threadIndex ] = fetchKernel( threadIdx.x, threadIdx.y, threadIdx.z );

   __syncthreads();

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

            Index threadIndex = kernelIndexX + kernelWidth * kernelIndexY + kernelWidth * kernelHeight * kernelIndexZ;

            if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY || elementIndexZ < 0
                || elementIndexZ >= endZ )
            {
               result = convolve( result,
                                  fetchBoundary( elementIndexX, elementIndexY, elementIndexZ ),
                                  shared[threadIndex] );
            }
            else {
               result = convolve( result,
                                  fetchData( elementIndexX, elementIndexY, elementIndexZ ),
                                  shared[threadIndex] );
            }
         }
      }
   }

   store( ix, iy, iz, result );
}

#endif
