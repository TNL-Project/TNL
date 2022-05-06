
#ifdef HAVE_CUDA

/**
 * This method stores image tile into shared memory
 * and then calculates convolution.
 *
 * Thanks for the idea https://www.evl.uic.edu/sjames/cs525/final.html
 */

#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/SharedMemory.h>

template< int Dimension, typename Device >
struct Convolution;

template< typename Index,
          typename Real,
          typename FetchData,
          typename FetchBoundary,
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
               Convolve convolve,
               Store store )
{
   Real* data = TNL::Cuda::getSharedMemory< Real >();

   const Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   const Index ix = threadIdx.x + blockIdx.x * blockDim.x;

   const Index radiusY = kernelHeight >> 1;
   const Index radiusX = kernelWidth >> 1;

   const Index dataBlockWidth = 2 * kernelWidth - 1;
   const Index dataBlockHeight = 2 * kernelHeight - 1;

   const Index dataBlockRadiusX = dataBlockWidth >> 1;
   const Index dataBlockRadiusY = dataBlockHeight >> 1;

   Index x, y, index;

   // Top Left
   x = ix - radiusX;
   y = iy - radiusY;
   index = threadIdx.x + threadIdx.y * dataBlockWidth;

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Top right
   x = ix + radiusX;
   y = iy - radiusY;
   index = dataBlockRadiusX + threadIdx.x + threadIdx.y * dataBlockWidth;

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Bottom Left
   x = ix - radiusX;
   y = iy + radiusY;
   index = threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth;

   if(x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Bottom Right
   x = ix + radiusX;
   y = iy + radiusY;
   index = dataBlockRadiusX + threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth;

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   __syncthreads();

   if( ix >= endX || iy >= endY )
      return;

   Real result = 0;

   for( Index j = 0; j < kernelHeight; j++ ) {
      Index align = ( j + threadIdx.y ) * dataBlockWidth;

      for( Index i = 0; i < kernelWidth; i++ ) {
         Index index = i + threadIdx.x + align;

         result = convolve( result, ix, iy, i, j, data[ index ]);
      }
   }

   store( ix, iy, result );
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
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1;

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

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
             typename Convolve,
             typename Store >
   static void
   execute( const Vector< Index >& dimensions,
            const Vector< Index >& kernelSize,
            FetchData&& fetchData,
            FetchBoundary&& fetchBoundary,
            Convolve&& convolve,
            Store&& store )
   {
      TNL::Cuda::LaunchConfiguration configuration;

      setup< Index, Real >( configuration, dimensions, kernelSize );

      constexpr auto kernel = convolution2D< Index, Real, FetchData, FetchBoundary, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       configuration,
                                       kernelSize.x(),
                                       kernelSize.y(),
                                       dimensions.x(),
                                       dimensions.y(),
                                       fetchData,
                                       fetchBoundary,
                                       convolve,
                                       store );
   };
};

#endif
