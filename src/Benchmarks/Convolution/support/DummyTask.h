
#pragma once

template< int Dimension, typename Device >
struct Convolution
{
   template< typename Index >
   using Vector = TNL::Containers::StaticVector< Dimension, Index >;

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
            Store&& store );
};

template< typename Index, typename Real, int Dimension, typename Device >
struct DummyTask;

template< typename Index, typename Real >
struct DummyTask< Index, Real, 1, TNL::Devices::Cuda >
{
public:
   static constexpr int Dimension = 1;
   using Device = TNL::Devices::Cuda;
   using Vector = TNL::Containers::StaticVector< Dimension, Index >;
   using ConstDataStore = typename TNL::Containers::Vector< Real, Device, Index >::ConstViewType;
   using DataStore = typename TNL::Containers::Vector< Real, Device, Index >::ViewType;
   using ConvolutionLauncher = Convolution< Dimension, Device >;

   static void
   exec( const Vector& dimensions, const Vector& kernelSize, ConstDataStore& input, DataStore& result, ConstDataStore& kernel, int boundaryValue = 1 )
   {
      auto fetchData = [ = ] __cuda_callable__( Index i )
      {
         return input[ i ];
      };

      auto fetchBoundary = [ = ] __cuda_callable__( Index i )
      {
         return boundaryValue;
      };

      auto fetchKernel = [ = ] __cuda_callable__( Index i )
      {
         return kernel[ i ];
      };

      auto convolve = [ = ] __cuda_callable__( Real result, Real data, Real kernel )
      {
         return result + data * kernel;
      };

      auto store = [ = ] __cuda_callable__( Index i, Real resultValue ) mutable
      {
         result[ i ] = resultValue;
      };

      ConvolutionLauncher::execute< Index, Real >( dimensions,
                                                   kernelSize,
                                                   std::forward< decltype( fetchData ) >( fetchData ),
                                                   std::forward< decltype( fetchBoundary ) >( fetchBoundary ),
                                                   std::forward< decltype( fetchKernel ) >( fetchKernel ),
                                                   std::forward< decltype( convolve ) >( convolve ),
                                                   std::forward< decltype( store ) >( store ) );
   }
};

template< typename Index, typename Real >
struct DummyTask< Index, Real, 2, TNL::Devices::Cuda >
{
public:
   static constexpr int Dimension = 2;
   using Device = TNL::Devices::Cuda;
   using Vector = TNL::Containers::StaticVector< Dimension, Index >;
   using ConstDataStore = typename TNL::Containers::Vector< Real, Device, Index >::ConstViewType;
   using DataStore = typename TNL::Containers::Vector< Real, Device, Index >::ViewType;
   using ConvolutionLauncher = Convolution< Dimension, Device >;

   static void
   exec( const Vector& dimensions, const Vector& kernelSize, ConstDataStore& input, DataStore& result, ConstDataStore& kernel, int boundaryValue = 1 )
   {
      auto fetchData = [ = ] __cuda_callable__( Index i, Index j )
      {
         auto index = i + j * dimensions.x();

         return input[ index ];
      };

      auto fetchBoundary = [ = ] __cuda_callable__( Index i, Index j )
      {
         return boundaryValue;
      };

      auto fetchKernel = [ = ] __cuda_callable__( Index i, Index j )
      {
         auto index = i + j * kernelSize.x();

         return kernel[ index ];
      };

      auto convolve = [ = ] __cuda_callable__( Real result, Real data, Real kernel )
      {
         return result + data * kernel;
      };

      auto store = [ = ] __cuda_callable__( Index i, Index j, Real resultValue ) mutable
      {
         auto index = i + j * dimensions.x();

         result[ index ] = resultValue;
      };

      ConvolutionLauncher::execute< Index, Real >( dimensions,
                                                   kernelSize,
                                                   std::forward< decltype( fetchData ) >( fetchData ),
                                                   std::forward< decltype( fetchBoundary ) >( fetchBoundary ),
                                                   std::forward< decltype( fetchKernel ) >( fetchKernel ),
                                                   std::forward< decltype( convolve ) >( convolve ),
                                                   std::forward< decltype( store ) >( store ) );
   }
};

template< typename Index, typename Real >
struct DummyTask< Index, Real, 3, TNL::Devices::Cuda >
{
public:
   static constexpr int Dimension = 3;
   using Device = TNL::Devices::Cuda;
   using Vector = TNL::Containers::StaticVector< Dimension, Index >;
   using ConstDataStore = typename TNL::Containers::Vector< Real, Device, Index >::ConstViewType;
   using DataStore = typename TNL::Containers::Vector< Real, Device, Index >::ViewType;
   using ConvolutionLauncher = Convolution< Dimension, Device >;

   static void
   exec( const Vector& dimensions, const Vector& kernelSize, ConstDataStore& input, DataStore& result, ConstDataStore& kernel, int boundaryValue = 1 )
   {
      auto fetchData = [ = ] __cuda_callable__( Index i, Index j, Index k )
      {
         auto index = i + j * dimensions.x() + k * dimensions.x() * dimensions.y();

         return input[ index ];
      };

      auto fetchBoundary = [ = ] __cuda_callable__( Index i, Index j, Index k )
      {
         return boundaryValue;
      };

      auto fetchKernel = [ = ] __cuda_callable__( Index i, Index j, Index k )
      {
         auto index = i + j * kernelSize.x() + k * kernelSize.x() * kernelSize.y();

         return kernel[ index ];
      };

      auto convolve = [ = ] __cuda_callable__( Real result, Real data, Real kernel )
      {
         return result + data * kernel;
      };

      auto store = [ = ] __cuda_callable__( Index i, Index j, Index k, Real resultValue ) mutable
      {
         auto index = i + j * dimensions.x() + k * dimensions.x() * dimensions.y();

         result[ index ] = resultValue;
      };

      ConvolutionLauncher::execute< Index, Real >( dimensions,
                                                   kernelSize,
                                                   std::forward< decltype( fetchData ) >( fetchData ),
                                                   std::forward< decltype( fetchBoundary ) >( fetchBoundary ),
                                                   std::forward< decltype( fetchKernel ) >( fetchKernel ),
                                                   std::forward< decltype( convolve ) >( convolve ),
                                                   std::forward< decltype( store ) >( store ) );
   }
};
