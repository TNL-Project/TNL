
#ifdef HAVE_CUDA

#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/LaunchHelpers.h>

template< int Dimension, typename Device >
struct Convolution;

template<>
struct Convolution< 1, TNL::Devices::Cuda >
{
public:
   template< typename Index >
   static size_t
   getDynamicSharedMemorySize( Index kernelWidth, Index endX )
   {
      return 0;
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
   Index ix =  threadIdx.x + blockIdx.x * blockDim.x;

   if (ix >= endX)
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

template<>
struct Convolution< 2, TNL::Devices::Cuda >
{
public:
   template< typename Index >
   static size_t
   getDynamicSharedMemorySize( Index kernelWidth, Index kernelHeight, Index endX, Index endY )
   {
      return 0;
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
   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   if (ix >= endX || iy >= endY)
      return;

   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Real result = 0;

   for( Index j = - radiusY; j <= radiusY; j++ ) {
      Index elementIndexY = j + iy;
      Index kernelIndexY = j + radiusY;

      for( Index i = - radiusX; i <= radiusX; i++ ) {
         Index elementIndexX = i + ix;
         Index kernelIndexX = i + radiusX;

         if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY ) {
            result = convolve( result, fetchBoundary( elementIndexX, elementIndexY ), fetchKernel ( kernelIndexX, kernelIndexY ) );
         }
         else {
            result = convolve( result, fetchData( elementIndexX, elementIndexY ), fetchKernel( kernelIndexX, kernelIndexY ) );
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
   static size_t
   getDynamicSharedMemorySize( Index kernelWidth, Index kernelHeight, Index kernelDepth, Index endX, Index endY, Index endZ )
   {
      return 0;
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
   Index iz = threadIdx.z + blockIdx.z * blockDim.z;
   Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   if (ix >= endX || iy >= endY || iz >= endZ)
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

            if( elementIndexX < 0 || elementIndexX >= endX || elementIndexY < 0 || elementIndexY >= endY || elementIndexZ < 0 || elementIndexZ >= endZ ) {
               result = convolve( result, fetchBoundary( elementIndexX, elementIndexY, elementIndexZ ), fetchKernel( kernelIndexX, kernelIndexY, kernelIndexZ ) );
            }
            else {
               result = convolve( result, fetchData( elementIndexX, elementIndexY, elementIndexZ ), fetchKernel( kernelIndexX, kernelIndexY, kernelIndexZ ) );
            }
         }
      }
   }

   store( ix, iy, iz, result );
}

#endif
