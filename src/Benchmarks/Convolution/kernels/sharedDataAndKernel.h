#pragma once

#ifdef HAVE_CUDA

   #include <TNL/Devices/Cuda.h>
   #include <TNL/Containers/StaticVector.h>
   #include <TNL/Cuda/LaunchHelpers.h>
   #include <TNL/Cuda/SharedMemory.h>

/**
 * This method stores kernel and data in the shared memory to reduce amount of loads.
 *
 * We can calculate the size of shared memory needed the next way:
 * 1. We need to store in shared memory:
 *      * for 1D -> (2 * kernelWidth) - 1 < 2 * kernelWidth
 *      * for 2D -> ( (2 * kernelWidth) - 1 ) * ( (2 * kernelHeight) - 1 ) < 4 * kernelWidth * kernelHeight
 *      * for 3D -> ( (2 * kernelWidth) - 1 ) * ( (2 * kernelHeight) - 1 ) * ( (2 * kernelDepth) - 1 ) < 8 * kernelWidth *
 * kernelHeight * kernelDepth
 * 2. We take into account, that the maximal block size is 1024, so the maximum volume of kernel is 1024.
 *    Then the maximal amount of shared memory is:
 *      * for 1D -> 2 * 1024 -> 2048 elements (Note, that even if we take long double (16B) we still can fit in the shared
 * memory)
 *      * for 2D -> 4 * 1024 -> 4096 elements
 *      * for 3D -> 8 * 1024 -> 8196 elements (Note, that if double takes 8 bytes, then we can't fit tile into shared memory,
 * because we have 64 KB of data)
 * 3. The last thing is, that even if we take 1D and 2D case we have enough space to store 1024 kernel element.
 *    Then the maximal amount of shared memory is:
 *      * for 1D -> 3 * 1024 -> can use long double, double, float
 *      * for 2D -> 5 * 1024 -> can use double, float
 *      * for 3D -> 9 * 1024 -> can use float
 */

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

   Index kernelOffset = 2 * kernelWidth;

   Real* data = TNL::Cuda::getSharedMemory< Real >();
   Real* kernel = data + kernelOffset;

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

   kernel[ threadIdx.x ] = fetchKernel( threadIdx.x );

   __syncthreads();

   if( ix >= endX )
      return;

   Real result = 0;

   #pragma unroll
   for( Index i = 0; i < kernelWidth; i++ ) {
      Index elementIndex = i + threadIdx.x;

      result = convolve( result, data[ elementIndex ], kernel[ i ] );
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

   Index kernelOffset = ( 2 * kernelWidth - 1 ) * ( 2 * kernelHeight - 1 );

   Real* data = TNL::Cuda::getSharedMemory< Real >();
   Real* kernel = data + kernelOffset;

   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index x, y, index;

   // Top Left
   x = ix - radiusX;
   y = iy - radiusY;

   index = threadIdx.x + threadIdx.y * blockDim.x;

   kernel[ index ] = fetchKernel( threadIdx.x, threadIdx.y );

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Top right
   x = ix + radiusX;
   y = iy - radiusY;

   index = kernelWidth + threadIdx.x + threadIdx.y * blockDim.x;

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Bottom Left
   x = ix - radiusX;
   y = iy + radiusY;

   index = threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x;

   if( x < 0 || y < 0 || x >= endX || y >= endY ) {
      data[ index ] = fetchBoundary( x, y );
   }
   else {
      data[ index ] = fetchData( x, y );
   }

   // Bottom Right
   x = ix + radiusX;
   y = iy + radiusY;

   index = kernelWidth + threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x;

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

   #pragma unroll
   for( Index j = 0; j < kernelHeight; j++ ) {
      Index elementAlign = ( j + threadIdx.y ) * blockDim.x;
      Index kernelAlign = j * blockDim.x;

   #pragma unroll
      for( Index i = 0; i < kernelWidth; i++ ) {
         Index elementIndex = i + threadIdx.x + elementAlign;
         Index kernelIndex = i + kernelAlign;

         result = convolve( result, data[ elementIndex ], kernel[ kernelIndex ] );
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

   Index kernelOffset = ( 2 * kernelWidth - 1 ) * ( 2 * kernelHeight - 1 ) * ( 2 * kernelDepth - 1 );

   Real* data = TNL::Cuda::getSharedMemory< Real >();
   Real* kernel = data + kernelOffset;

   Index radiusZ = kernelDepth >> 1;
   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index x, y, z, index;

   // Z: 0 Y: 0 X: 0
   x = ix - radiusX;
   y = iy - radiusY;
   z = iz - radiusZ;

   index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

   kernel[ index ] = fetchKernel( threadIdx.x, threadIdx.y, threadIdx.z );

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

   index = kernelWidth + threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

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

   index = kernelWidth + threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

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

   index = threadIdx.x + threadIdx.y * blockDim.x + ( kernelDepth + threadIdx.z ) * blockDim.x * blockDim.y;

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

   index = kernelWidth + threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

   if( x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 0 X: 1
   x = ix + radiusX;
   y = iy - radiusY;
   z = iz + radiusZ;

   index = kernelWidth + threadIdx.x + threadIdx.y * blockDim.x + ( kernelDepth + threadIdx.z ) * blockDim.x * blockDim.y;

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

   index = threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x + ( kernelDepth + threadIdx.z ) * blockDim.x * blockDim.y;

   if(x < 0 || y < 0 || z < 0 || x >= endX || y >= endY || z >= endZ ) {
      data[ index ] = fetchBoundary( x, y, z );
   }
   else {
      data[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 1 X: 1
   x = ix + radiusX;
   y = iy + radiusY;
   z = iz + radiusZ;

   index = kernelWidth + threadIdx.x + ( kernelHeight + threadIdx.y ) * blockDim.x + ( kernelDepth + threadIdx.z ) * blockDim.x * blockDim.y;

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

   #pragma unroll
   for( Index k = 0; k < kernelDepth; k++ ) {
      Index xyAlign = ( k + threadIdx.z ) * blockDim.y * blockDim.x;
      Index xyKernelAlign = k * blockDim.x * blockDim.y;
   #pragma unroll
      for( Index j = 0; j < kernelHeight; j++ ) {
         Index xAlign = ( j + threadIdx.y ) * blockDim.x;
         Index xKernelAlign = j * blockDim.x;
   #pragma unroll
         for( Index i = 0; i < kernelWidth; i++ ) {
            Index elementIndex = i + threadIdx.x + xAlign + xyAlign;
            Index kernelIndex = i + xKernelAlign + xyKernelAlign;

            result = convolve( result, data[ elementIndex ], kernel[ kernelIndex ] );
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
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1;

      configuration.dynamicSharedMemorySize = ( kernelSize.x() + kernelElementCount ) * sizeof( Real );

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
      Index kernelVolume = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ ) {
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1;
         kernelVolume *= kernelSize[ i ];
      }

      configuration.dynamicSharedMemorySize = ( kernelVolume + kernelElementCount ) * sizeof( Real );

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
      Index kernelVolume = 1;

      for( Index i = 0; i < kernelSize.getSize(); i++ ) {
         kernelElementCount *= ( 2 * kernelSize[ i ] ) - 1;
         kernelVolume *= kernelSize[ i ];
      }

      configuration.dynamicSharedMemorySize = ( kernelVolume + kernelElementCount ) * sizeof( Real );

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
