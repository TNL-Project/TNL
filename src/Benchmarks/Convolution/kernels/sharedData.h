#pragma once

#ifdef HAVE_CUDA

/**
 * This method stores image tile into shared memory
 * and then calculates convolution.
 *
 * Thanks for the idea  https://www.evl.uic.edu/sjames/cs525/final.html
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

   if( ix >= endX )
      return;

   Real* shared = TNL::Cuda::getSharedMemory< Real >();
   Index radius = kernelWidth >> 1;

   // Left
   Index lhs = ix - radius;

   if( lhs < 0 ) {
      shared[ threadIdx.x ] = fetchBoundary( lhs );
   }
   else {
      shared[ threadIdx.x ] = fetchData( lhs );
   }

   // Right
   Index rhs = ix + radius;

   if( rhs >= endX ) {
      shared[ threadIdx.x + blockDim.x ] = fetchBoundary( rhs );
   }
   else {
      shared[ threadIdx.x + blockDim.x ] = fetchData( rhs );
   }

   __syncthreads();

   Real result = 0;

   #pragma unroll
   for( Index i = 0; i < kernelWidth; i++ ) {
      Index elementIndex = i + threadIdx.x;

      result = convolve( result, shared[ elementIndex ], fetchKernel( i ) );
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

   Real* shared = TNL::Cuda::getSharedMemory< Real >();

   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index x, y, index;

   // Top Left
   x = ix - radiusX;
   y = iy - radiusY;

   index = threadIdx.x + threadIdx.y * blockDim.x;

   if( x < 0 || y < 0 ) {
      shared[ index ] = fetchBoundary( x, y );
   }
   else {
      shared[ index ] = fetchData( x, y );
   }

   // Top right
   x = ix + radiusX;
   y = iy - radiusY;

   index = radiusX + threadIdx.x + threadIdx.y * blockDim.x;

   if( x >= endX || y < 0 ) {
      shared[ index ] = fetchBoundary( x, y );
   }
   else {
      shared[ index ] = fetchData( x, y );
   }

   // Bottom Left
   x = ix - radiusX;
   y = iy + radiusY;

   index = threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.x;

   if( x < 0 || y >= endY ) {
      shared[ index ] = fetchBoundary( x, y );
   }
   else {
      shared[ index ] = fetchData( x, y );
   }

   // Bottom Right
   x = ix + radiusX;
   y = iy + radiusY;

   index = radiusX + threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.x;

   if( x >= endX || y >= endY ) {
      shared[ index ] = fetchBoundary( x, y );
   }
   else {
      shared[ index ] = fetchData( x, y );
   }

   __syncthreads();

   Real result = 0;

   for( Index j = 0; j <= radiusY; j++ ) {
      Index align = ( j + threadIdx.y ) * blockDim.y;

      for( Index i = 0; i <= radiusX; i++ ) {
         Index index = i + threadIdx.x + align;

         result = convolve( result, shared[ index ], fetchKernel( i, j ) );
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

   Real* shared = TNL::Cuda::getSharedMemory< Real >();

   Index radiusZ = kernelDepth >> 1;
   Index radiusY = kernelHeight >> 1;
   Index radiusX = kernelWidth >> 1;

   Index x, y, z, index;

   // Z: 0 Y: 0 X: 0
   x = ix - radiusX;
   y = iy - radiusY;
   z = iz - radiusZ;

   index = threadIdx.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.x * blockDim.y;

   if( x < 0 || y < 0 || z < 0 ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 0 X: 1
   x = ix + radiusX;
   y = iy - radiusY;
   z = iz - radiusZ;

   index = radiusX + threadIdx.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.x * blockDim.y;

   if( x >= endX || y < 0 || z < 0 ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 1 X: 0
   x = ix - radiusX;
   y = iy + radiusY;
   z = iz - radiusZ;

   index = radiusX + threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.y + threadIdx.z * blockDim.x * blockDim.y;

   if( x < 0 || y >= endY || z < 0 ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 0 X: 0
   x = ix - radiusX;
   y = iy - radiusY;
   z = iz + radiusZ;

   index = threadIdx.x + threadIdx.y * blockDim.y + ( radiusZ + threadIdx.z ) * blockDim.x * blockDim.y;

   if( x < 0 || y < 0 || z >= endZ ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 0 Y: 1 X: 1
   x = ix + radiusX;
   y = iy + radiusY;
   z = iz - radiusZ;

   index = radiusX + threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.y + threadIdx.z * blockDim.x * blockDim.y;

   if( x >= endX || y >= endY || z < 0 ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 0 X: 1
   x = ix + radiusX;
   y = iy - radiusY;
   z = iz + radiusZ;

   index = radiusX + threadIdx.x + threadIdx.y * blockDim.y + ( radiusZ + threadIdx.z ) * blockDim.x * blockDim.y;

   if( x >= endX || y < 0 || z >= endZ ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 1 X: 0
   x = ix - radiusX;
   y = iy + radiusY;
   z = iz + radiusZ;

   index = threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.y + ( radiusZ + threadIdx.z ) * blockDim.x * blockDim.y;

   if( x < 0 || y >= endY || z >= endZ ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   // Z: 1 Y: 1 X: 1
   x = ix + radiusX;
   y = iy + radiusY;
   z = iz + radiusZ;

   index = radiusX + threadIdx.x + ( radiusY + threadIdx.y ) * blockDim.y + ( radiusZ + threadIdx.z ) * blockDim.x * blockDim.y;

   if( x >= endX || y >= endY || z >= endZ ) {
      shared[ index ] = fetchBoundary( x, y, z );
   }
   else {
      shared[ index ] = fetchData( x, y, z );
   }

   __syncthreads();

   Real result = 0;

   for( Index k = 0; k <= radiusZ; k++ ) {
      Index xyAlign = ( k + threadIdx.z ) * blockDim.y * blockDim.x;

      for( Index j = 0; j <= radiusY; j++ ) {
         Index xAlign = ( j + threadIdx.y ) * blockDim.y;

         for( Index i = 0; i <= radiusX; i++ ) {
            Index index = i + threadIdx.x + xAlign + xyAlign;

            result = convolve( result, shared[ index ], fetchKernel( i, j, k ) );
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
      configuration.gridSize.y =
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
