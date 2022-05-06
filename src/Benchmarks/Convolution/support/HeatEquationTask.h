
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
             typename Convolve,
             typename Store >
   static void
   execute( const Vector< Index >& dimensions,
            const Vector< Index >& kernelSize,
            FetchData&& fetchData,
            FetchBoundary&& fetchBoundary,
            Convolve&& convolve,
            Store&& store );
};

template< typename Index, typename Real, int Dimension, typename Device >
struct HeatEquationTask;

template< typename Index, typename Real >
struct HeatEquationTask< Index, Real, 2, TNL::Devices::Cuda >
{
public:
   static constexpr int Dimension = 2;
   using Device = TNL::Devices::Cuda;
   using Vector = TNL::Containers::StaticVector< Dimension, Index >;
   using Point = TNL::Containers::StaticVector< Dimension, Real >;
   using ConstDataStore = typename TNL::Containers::Vector< Real, Device, Index >::ConstViewType;
   using DataStore = typename TNL::Containers::Vector< Real, Device, Index >::ViewType;
   using ConvolutionLauncher = Convolution< Dimension, Device >;

   static void
   exec( const Vector& dimensions,
         const Vector& kernelSize,
         const Point& functionDomain,
         const Point& kernelDomain,
         const Real time,
         ConstDataStore& input,
         DataStore& result)
   {
      auto functionSpaceSteps = Point(functionDomain.x() / dimensions.x(), functionDomain.y() / dimensions.y());
      auto kernelSpaceSteps = Point(kernelDomain.x() / kernelSize.x(), kernelDomain.y() / kernelSize.y());

      auto fetchData = [ = ] __cuda_callable__( Index i, Index j )
      {
         auto index = i + j * dimensions.x();

         return input[ index ];
      };

      auto fetchBoundary = [ = ] __cuda_callable__( Index i, Index j )
      {
         return 0;
      };

      auto convolve = [ = ] __cuda_callable__( Real result, Index dataX, Index dataY, Index kernelX, Index kernelY, Real data )
      {
         auto functionXPos = dataX * functionSpaceSteps.x() - (functionDomain.x() / 2),
              functionYPos = dataY * functionSpaceSteps.y() - (functionDomain.y() / 2);

         auto kernelXPos = (kernelX - kernelSize.x() / 2) * kernelSpaceSteps.x(),
              kernelYPos = (kernelY - kernelSize.y() / 2) * kernelSpaceSteps.y();

         auto deltaXPos = kernelXPos - functionXPos,
              deltaYPos = kernelYPos - functionYPos;

         auto kernel = kernelSpaceSteps.x() * kernelSpaceSteps.y() * ( (Real)1 / ( (Real)4 * M_PI * time ) ) * exp( - ( pow(deltaXPos, 2.) + pow(deltaYPos, 2.)  ) / ( (Real)4 * time ) );

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
                                                   std::forward< decltype( convolve ) >( convolve ),
                                                   std::forward< decltype( store ) >( store ) );
   }
};
