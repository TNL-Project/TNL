
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

// template<>
// struct Convolution< 2, TNL::Devices::Cuda >
// {
// public:
//    template< typename Index >
//    static size_t
//    getDynamicSharedMemorySize( Index kernelWidth, Index kernelHeight, Index endX, Index endY )
//    {
//       return 0;
//    }
// };

// template< typename Index,
//           typename Real,
//           typename FetchData,
//           typename FetchBoundary,
//           typename FetchKernel,
//           typename Convolve,
//           typename Store >
// __global__
// static void
// convolution2D( Index kernelWidth,
//                Index kernelHeight,
//                Index endX,
//                Index endY,
//                FetchData& fetchData,
//                FetchBoundary& fetchBoundary,
//                FetchKernel& fetchKernel,
//                Convolve& convolve,
//                Store& store )
// {
//    int iy = threadIdx.y + blockIdx.y * blockDim.y;
//    int ix = threadIdx.x + blockIdx.x * blockDim.x;

//    Real result = 0;

//    for( Index j = iy - kernelHeight; j <= iy + kernelHeight; j++ ) {
//       for( Index i = ix - kernelWidth; i <= ix + kernelWidth; i++ ) {
//          if( i < 0 || i >= endX || j < 0 || j >= endY ) {
//             result = convolve( result, fetchBoundary( i, j ) );
//          }
//          else {
//             result = convolve( result, fetchData( i, j ), fetchKernel( i, j ) );
//          }
//       }
//    }

//    store( ix, iy, result );
// }

// template<>
// struct Convolution< 3, TNL::Devices::Cuda >
// {
// public:
//    template< typename Index >
//    static size_t
//    getDynamicSharedMemorySize( Index kernelWidth, Index kernelHeight, Index kernelDepth, Index endX, Index endY, Index endZ )
//    {
//       return 0;
//    }
// };

// template< typename Index,
//           typename Real,
//           typename FetchData,
//           typename FetchBoundary,
//           typename FetchKernel,
//           typename Convolve,
//           typename Store >
// __global__
// static void
// convolution3D( Index kernelWidth,
//                Index kernelHeight,
//                Index kernelDepth,
//                Index endX,
//                Index endY,
//                Index endZ,
//                FetchData& fetchData,
//                FetchBoundary& fetchBoundary,
//                FetchKernel& fetchKernel,
//                Convolve& convolve,
//                Store& store )
// {
//    int ix = threadIdx.x + blockIdx.x * blockDim.x;
//    int iy = threadIdx.y + blockIdx.y * blockDim.y;
//    int iz = threadIdx.z + blockIdx.z * blockDim.z;

//    Real result = 0;

//    for( Index k = iz - kernelDepth; k <= iz + kernelDepth; k++ ) {
//       for( Index j = iy - kernelHeight; j <= iy + kernelHeight; j++ ) {
//          for( Index i = ix - kernelWidth; i <= ix + kernelWidth; i++ ) {
//             if( i < 0 || i >= endX || j < 0 || j >= endY || k < 0 || k >= endZ ) {
//                result = convolve( result, fetchBoundary( i, j, k ) );
//             }
//             else {
//                result = convolve( result, fetchData( i, j, k ), fetchKernel( i, j, k ) );
//             }
//          }
//       }
//    }

//    store( ix, iy, iz, result );
// }

#endif
