
#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Cuda/KernelLaunch.h>

template< int Dimension, typename Device >
struct Convolution;

template< int Dimension, typename Device >
struct Launcher;

template<>
struct Launcher< 1, TNL::Devices::Cuda >
{
public:
   using Vector = TNL::Containers::StaticVector< 1, int >;
   using ConvolutionKernel = Convolution< 1, TNL::Devices::Cuda >;

   template< typename Index, typename Real, typename FetchData, typename FetchBoundary, typename FetchKernel, typename Convolve, typename Store >
   static inline void
   exec( const Vector& dimensions,
         const Vector& kernelSize,
         FetchData&& fetchData,
         FetchBoundary&& fetchBoundary,
         FetchKernel&& fetchKernel,
         Convolve&& convolve,
         Store&& store )
   {
      TNL::Cuda::LaunchConfiguration launchConfig;

      launchConfig.dynamicSharedMemorySize =
         ConvolutionKernel::getDynamicSharedMemorySize< Index >( kernelSize.x(), dimensions.x() );

      // TODO: - Benchmark the best value
      launchConfig.blockSize.x = 256;
      launchConfig.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( dimensions.x(), launchConfig.blockSize.x ) );

      if( (std::size_t) launchConfig.blockSize.x * launchConfig.gridSize.x < (std::size_t) dimensions.x() ) {
         const int desGridSize = 32 * TNL::Cuda::DeviceInfo::getCudaMultiprocessors( TNL::Cuda::DeviceInfo::getActiveDevice() );

         launchConfig.gridSize.x =
            TNL::min( desGridSize, TNL::Cuda::getNumberOfBlocks( dimensions.x(), launchConfig.blockSize.x ) );
      }

      constexpr auto kernel = convolution1D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       launchConfig,
                                       kernelSize.x(),
                                       dimensions.x(),
                                       fetchData,
                                       fetchBoundary,
                                       fetchKernel,
                                       convolve,
                                       store );
   }
};

template<>
struct Launcher< 2, TNL::Devices::Cuda >
{
public:
   using Vector = TNL::Containers::StaticVector< 2, int >;
   using ConvolutionKernel = Convolution< 2, TNL::Devices::Cuda >;

   template< typename Index, typename Real, typename FetchData, typename FetchBoundary, typename FetchKernel, typename Convolve, typename Store >
   static inline void
   exec( const Vector& dimensions,
         const Vector& kernelSize,
         FetchData&& fetchData,
         FetchBoundary&& fetchBoundary,
         FetchKernel&& fetchKernel,
         Convolve&& convolve,
         Store&& store )
   {
      TNL::Cuda::LaunchConfiguration launchConfig;

      launchConfig.dynamicSharedMemorySize = ConvolutionKernel::getDynamicSharedMemorySize< Index >(
         kernelSize.x(), kernelSize.y(), dimensions.x(), dimensions.y() );

      const Index sizeX = dimensions.x();
      const Index sizeY = dimensions.y();

      if( sizeX >= sizeY * sizeY ) {
         launchConfig.blockSize.x = TNL::min( 256, sizeX );
         launchConfig.blockSize.y = 1;
      }
      else if( sizeY >= sizeX * sizeX ) {
         launchConfig.blockSize.x = 1;
         launchConfig.blockSize.y = TNL::min( 256, sizeY );
      }
      else {
         launchConfig.blockSize.x = TNL::min( 32, sizeX );
         launchConfig.blockSize.y = TNL::min( 8, sizeY );
      }

      launchConfig.gridSize.x =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( sizeX, launchConfig.blockSize.x ) );
      launchConfig.gridSize.y =
         TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( sizeY, launchConfig.blockSize.y ) );

      constexpr auto kernel = convolution2D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       launchConfig,
                                       kernelSize.x(),
                                       kernelSize.y(),
                                       dimensions.x(),
                                       dimensions.y(),
                                       fetchData,
                                       fetchBoundary,
                                       fetchKernel,
                                       convolve,
                                       store );
   }
};

// template<>
// struct Launcher< 3, TNL::Devices::Cuda >
// {
// public:
//    using Vector = TNL::Containers::StaticVector< 3, int >;
//    using ConvolutionKernel = Convolution< 3, TNL::Devices::Cuda >;

//    template< typename Index, typename Real, typename FetchData, typename FetchBoundary, typename FetchKernel, typename Convolve, typename Store >
//    static inline void
//    exec( const Vector& dimensions,
//          const Vector& kernelSize,
//          FetchData&& fetchData,
//          FetchBoundary&& fetchBoundary,
//          FetchKernel&& fetchKernel,
//          Convolve&& convolve,
//          Store&& store )
//    {
//       const Index sizeX = dimensions.x();
//       const Index sizeY = dimensions.y();
//       const Index sizeZ = dimensions.z();

//       TNL::Cuda::LaunchConfiguration launchConfig;

//       launchConfig.dynamicSharedMemorySize = ConvolutionKernel::getDynamicSharedMemorySize< Index >(
//          kernelSize.x(), kernelSize.y(), kernelSize.z(), dimensions.x(), dimensions.y(), dimensions.z() );

//       if( sizeX >= sizeY * sizeY * sizeZ * sizeZ ) {
//          launchConfig.blockSize.x = TNL::min( 256, sizeX );
//          launchConfig.blockSize.y = 1;
//          launchConfig.blockSize.z = 1;
//       }
//       else if( sizeY >= sizeX * sizeX * sizeZ * sizeZ ) {
//          launchConfig.blockSize.x = 1;
//          launchConfig.blockSize.y = TNL::min( 256, sizeY );
//          launchConfig.blockSize.z = 1;
//       }
//       else if( sizeZ >= sizeX * sizeX * sizeY * sizeY ) {
//          launchConfig.blockSize.x = TNL::min( 2, sizeX );
//          launchConfig.blockSize.y = TNL::min( 2, sizeY );
//          // CUDA allows max 64 for launchConfig.blockSize.z
//          launchConfig.blockSize.z = TNL::min( 64, sizeZ );
//       }
//       else if( sizeX >= sizeZ * sizeZ && sizeY >= sizeZ * sizeZ ) {
//          launchConfig.blockSize.x = TNL::min( 32, sizeX );
//          launchConfig.blockSize.y = TNL::min( 8, sizeY );
//          launchConfig.blockSize.z = 1;
//       }
//       else if( sizeX >= sizeY * sizeY && sizeZ >= sizeY * sizeY ) {
//          launchConfig.blockSize.x = TNL::min( 32, sizeX );
//          launchConfig.blockSize.y = 1;
//          launchConfig.blockSize.z = TNL::min( 8, sizeZ );
//       }
//       else if( sizeY >= sizeX * sizeX && sizeZ >= sizeX * sizeX ) {
//          launchConfig.blockSize.x = 1;
//          launchConfig.blockSize.y = TNL::min( 32, sizeY );
//          launchConfig.blockSize.z = TNL::min( 8, sizeZ );
//       }
//       else {
//          launchConfig.blockSize.x = TNL::min( 16, sizeX );
//          launchConfig.blockSize.y = TNL::min( 4, sizeY );
//          launchConfig.blockSize.z = TNL::min( 4, sizeZ );
//       }
//       launchConfig.gridSize.x =
//          TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( sizeX, launchConfig.blockSize.x ) );
//       launchConfig.gridSize.y =
//          TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( sizeY, launchConfig.blockSize.y ) );
//       launchConfig.gridSize.z =
//          TNL::min( TNL::Cuda::getMaxGridSize(), TNL::Cuda::getNumberOfBlocks( sizeZ, launchConfig.blockSize.z ) );

//       dim3 gridCount;
//       gridCount.x = roundUpDivision( sizeX, launchConfig.blockSize.x * launchConfig.gridSize.x );
//       gridCount.y = roundUpDivision( sizeY, launchConfig.blockSize.y * launchConfig.gridSize.y );
//       gridCount.z = roundUpDivision( sizeZ, launchConfig.blockSize.z * launchConfig.gridSize.z );

//       constexpr auto kernel = convolution3D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

//       TNL::Cuda::launchKernel< true >( kernel,
//                                        0,
//                                        launchConfig,
//                                        kernelSize.x(),
//                                        kernelSize.y(),
//                                        kernelSize.z(),
//                                        dimensions.x(),
//                                        dimensions.y(),
//                                        dimensions.z(),
//                                        std::forward< FetchData >( fetchData ),
//                                        std::forward< FetchBoundary >( fetchBoundary ),
//                                        std::forward< FetchKernel >( fetchKernel ),
//                                        std::forward< Convolve >( convolve ),
//                                        std::forward< Store >( store ) );
//    }
// };
