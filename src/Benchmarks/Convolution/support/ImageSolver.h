
#pragma once

#include "Solver.h"
#include "DummyTask.h"

#include <TNL/FileName.h>
#include <TNL/Timer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Images/PNGImage.h>
#include <TNL/Images/RegionOfInterest.h>

static std::vector< TNL::String > dimensionIds = { "x-dimension", "y-dimension", "z-dimension" };
static std::vector< TNL::String > kernelSizeIds = { "x-kernel-size", "y-kernel-size", "z-kernel-size" };

class ImageSolver : public Solver< 2, TNL::Devices::Cuda >
{
public:
   constexpr static int Dimension = 2;
   using Device = TNL::Devices::Cuda;

   using Base = Solver< Dimension, Device >;
   using Vector = TNL::Containers::StaticVector< Dimension, int >;
   using DataStore = TNL::Containers::Vector< float, Device, int >;
   using HostDataStore = TNL::Containers::Vector< float, TNL::Devices::Host, int >;

   using GridType = TNL::Meshes::Grid< 2, float, Device, int >;
   using GridPointer = TNL::Pointers::SharedPointer< GridType >;
   using MeshFunctionType = TNL::Functions::MeshFunction< GridType >;

   virtual void
   start( const TNL::Config::ParameterContainer& parameters ) const override
   {
      GridPointer grid;
      MeshFunctionType meshFunction;
      TNL::Images::PNGImage< int > image;
      TNL::Images::RegionOfInterest< int > roi;

      meshFunction.setMesh( grid );

      auto output = parameters.getParameter< TNL::String >( "output" );

      if (!this -> readImage(parameters, grid, meshFunction, image, roi) ||
          !this -> convolve(parameters, meshFunction) ||
          !this -> write(parameters, image, meshFunction))
         return;
   }

   template<typename Image>
   bool readImage(const TNL::Config::ParameterContainer& parameters,
                  GridPointer & grid,
                  MeshFunctionType& meshFunction,
                  Image& image,
                  TNL::Images::RegionOfInterest< int >& roi) const {
      auto input = parameters.getParameter< TNL::String >( "input" );

      if( image.openForRead( input ) ) {
         if( ! roi.setup( parameters, &image ) ) {
            std::cout << "Invalid image roi.";
            image.close();
            return false;
         }

         std::cout << image.getWidth() << " " << image.getHeight() << std::endl;

         auto meshPointer = meshFunction.getMeshPointer();

         meshPointer -> setDimensions(image.getWidth(), image.getHeight());

         meshFunction.setMesh(meshPointer);

         if( ! image.read( roi, meshFunction ) ) {
            std::cout << "Invalid image size" << std::endl;;
            image.close();
            return false;
         }

         image.close();

         std::cout << "Image read was successful: " << meshFunction.getData().getSize() << " elements count" << std::endl;
         return true;
      }

      std::cout << "Image open for read failed. Please check file path" << std::endl;;

      return false;
   }

   bool convolve(const TNL::Config::ParameterContainer& parameters, MeshFunctionType& meshFunction) const {
      auto imageData = meshFunction.getData().getConstView();

      Vector kernelSize;
      DataStore kernel;

      kernel = getKernel(parameters, kernelSize);

      DataStore result;

      result.setLike( imageData );
      result = 0;

      TNL::Timer timer;

      timer.start();

      std::cout << imageData.getSize() << " " << result.getSize() << std::endl;

      launchConvolution( imageData,
                         kernel.getConstView(),
                         result.getView(),
                         meshFunction.getMeshPointer() -> getDimensions(),
                         kernelSize );

      timer.stop();

      meshFunction.getData() = result;

      std::cout << "Image convolution was successful. Time: " << timer.getRealTime() << " sec" << std::endl;

      return true;
   }

   template<typename Image>
   bool write(const TNL::Config::ParameterContainer& parameters, Image& image, MeshFunctionType& meshFunction) const {
      auto output = parameters.getParameter< TNL::String >( "output" );
      GridType grid = meshFunction.getMesh();

      if( image.openForWrite( output, grid ) ) {
         if( ! image.write( meshFunction ) ) {
            std::cout << "Image write failed" << std::endl;;
            image.close();
            return false;
         }

         image.close();

         return true;
      }

      std::cout << "Image open for write failed. Please check file path" << std::endl;

      return false;
   }

   HostDataStore getKernel( const TNL::Config::ParameterContainer& parameters, Vector& kernelDimension ) const {
      kernelDimension = {3, 3};

      return {-1, -1, -1,
              -1, 8, -1,
              -1, -1, -1};
   }

   void
   launchConvolution( DataStore::ConstViewType image,
                      DataStore::ConstViewType kernel,
                      DataStore::ViewType result,
                      const GridType::CoordinatesType& imageDimension,
                      const GridType::CoordinatesType& kernelDimension) const
   {
      DummyTask<int, float, Dimension, Device>::exec(imageDimension, kernelDimension, image, result, kernel);
   }

   virtual TNL::Config::ConfigDescription
   makeInputConfig() const override
   {
      TNL::Config::ConfigDescription config = Base::makeInputConfig();

      config.addDelimiter( "Image settings:" );

      config.addEntry< TNL::String >( "input", "PNG image" );
      config.addEntry< TNL::String >( "output", "PNG image" );

      config.addDelimiter( "Roi settings:" );

      config.addEntry< int >( "roi-top", "Top (smaller number) line of the region of interest.", -1 );
      config.addEntry< int >( "roi-bottom", "Bottom (larger number) line of the region of interest.", -1 );
      config.addEntry< int >( "roi-left", "Left (smaller number) column of the region of interest.", -1 );
      config.addEntry< int >( "roi-right", "Right (larger number) column of the region of interest.", -1 );

      return config;
   }
};
