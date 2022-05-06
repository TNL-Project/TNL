#pragma once

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

   Real* data = TNL::Cuda::getSharedMemory< Real >();
   Index radius = kernelWidth >> 1;

   // Left
   Index lhs = ix - radius;

   if( lhs < 0 || lhs >= endX ) {
      data[ threadIdx.x ] = fetchBoundary( lhs );
   }
   else {
      data[ threadIdx.x ] = fetchData( lhs );
   }

   // Right
   Index rhs = ix + radius;

   if( rhs < 0 || rhs >= endX ) {
      data[ threadIdx.x + blockDim.x ] = fetchBoundary( rhs );
   }
   else {
      data[ threadIdx.x + blockDim.x ] = fetchData( rhs );
   }

   __syncthreads();

   if( ix >= endX )
      return;

   Real result = 0;

   #pragma unroll
   for( Index i = 0; i < kernelWidth; i++ ) {
      Index elementIndex = i + threadIdx.x;

      result = convolve( result, data[ elementIndex ], fetchKernel( i ) );
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

         result = convolve( result, data[ index ], fetchKernel( i, j ) );
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
   Real* data = TNL::Cuda::getSharedMemory< Real >();

   const Index ix = threadIdx.x + blockIdx.x * blockDim.x;
   const Index iy = threadIdx.y + blockIdx.y * blockDim.y;
   const Index iz = threadIdx.z + blockIdx.z * blockDim.z;

   const Index radiusX = kernelWidth >> 1;
   const Index radiusY = kernelHeight >> 1;
   const Index radiusZ = kernelDepth >> 1;

   const Index dataBlockWidth = 2 * kernelWidth - 1;
   const Index dataBlockHeight = 2 * kernelHeight - 1;
   const Index dataBlockDepth = 2 * kernelDepth - 1;

   const Index dataBlockXYVolume = dataBlockWidth * dataBlockHeight;

   const Index dataBlockRadiusX = dataBlockWidth >> 1;
   const Index dataBlockRadiusY = dataBlockHeight >> 1;
   const Index dataBlockRadiusZ = dataBlockDepth >> 1;

   Index x, y, z, index;

   // Z: 0 Y: 0 X: 0
   x = ix - radiusX;
   y = iy - radiusY;
   z = iz - radiusZ;

   index = threadIdx.x + threadIdx.y * dataBlockWidth + threadIdx.z * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 0 X: 1
   x = ix + radiusX;
   y = iy - radiusY;
   z = iz - radiusZ;

   index = dataBlockRadiusX + threadIdx.x + threadIdx.y * dataBlockWidth + threadIdx.z * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 1 X: 0
   x = ix - radiusX;
   y = iy + radiusY;
   z = iz - radiusZ;

   index = dataBlockRadiusX + threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth + threadIdx.z * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 0 X: 0
   x = ix - radiusX;
   y = iy - radiusY;
   z = iz + radiusZ;

   index = threadIdx.x + threadIdx.y * dataBlockWidth + ( dataBlockRadiusZ + threadIdx.z ) * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 1 X: 1
   x = ix + radiusX;
   y = iy + radiusY;
   z = iz - radiusZ;

   index = dataBlockRadiusX + threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth + threadIdx.z * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 0 X: 1
   x = ix + radiusX;
   y = iy - radiusY;
   z = iz + radiusZ;

   index = dataBlockRadiusX + threadIdx.x + threadIdx.y * dataBlockWidth + ( dataBlockRadiusZ + threadIdx.z ) * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 1 X: 0
   x = ix - radiusX;
   y = iy + radiusY;
   z = iz + radiusZ;

   index = threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth + ( dataBlockRadiusZ + threadIdx.z ) * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 1 X: 1
   x = ix + radiusX;
   y = iy + radiusY;
   z = iz + radiusZ;

   index = dataBlockRadiusX + threadIdx.x + ( dataBlockRadiusY + threadIdx.y ) * dataBlockWidth + ( dataBlockRadiusZ + threadIdx.z ) * dataBlockXYVolume;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   __syncthreads();

   if( ix >= endX || iy >= endY || iz >= endZ )
      return;

   Real result = 0;

   for( Index k = 0; k < kernelDepth; k++ ) {
      Index xyAlign = ( k + threadIdx.z ) * dataBlockXYVolume;

      for( Index j = 0; j < kernelHeight; j++ ) {
         Index xAlign = ( j + threadIdx.y ) * dataBlockWidth;

         for( Index i = 0; i < kernelWidth; i++ ) {
            Index index = i + threadIdx.x + xAlign + xyAlign;

            result = convolve( result, data[ index ], fetchKernel( i, j, k ) );
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
      Index kernelElementCount = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ )
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1 ;

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

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
      Index kernelElementCount = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ )
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1;

      configuration.dynamicSharedMemorySize = kernelElementCount * sizeof( Real );

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
