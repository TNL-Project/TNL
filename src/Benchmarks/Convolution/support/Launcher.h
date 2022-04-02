
#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Cuda/KernelLaunch.h>

template< int Dimension, typename Device >
struct Convolution {
   template< typename Index >
   using Vector = TNL::Containers::StaticVector< 1, Index >;

   template< typename Index, typename Real >
   static void
   setup(TNL::Cuda::LaunchConfiguration& configuration, const Vector< Index >& dimensions, const Vector< Index >& kernelSize);
};

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

      ConvolutionKernel::setup<Index, Real>(launchConfig, dimensions, kernelSize);

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

      ConvolutionKernel::setup<Index, Real>(launchConfig, dimensions, kernelSize);

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

template<>
struct Launcher< 3, TNL::Devices::Cuda >
{
public:
   using Vector = TNL::Containers::StaticVector< 3, int >;
   using ConvolutionKernel = Convolution< 3, TNL::Devices::Cuda >;

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
      const Index sizeX = dimensions.x();
      const Index sizeY = dimensions.y();
      const Index sizeZ = dimensions.z();

      TNL::Cuda::LaunchConfiguration launchConfig;

      ConvolutionKernel::setup<Index, Real>(launchConfig, dimensions, kernelSize);

      constexpr auto kernel = convolution3D< Index, Real, FetchData, FetchBoundary, FetchKernel, Convolve, Store >;

      TNL::Cuda::launchKernel< true >( kernel,
                                       0,
                                       launchConfig,
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
   }
};
