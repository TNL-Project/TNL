/***************************************************************************
                          CompressibleConservativeVariables.h  -  description
                             -------------------
    begin                : Feb 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {

template< typename Mesh >
class CompressibleConservativeVariables
{
   public:
      typedef Mesh MeshType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;      
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VelocityFieldType > MomentumFieldPointer;
      
      CompressibleConservativeVariables(){};
      
      CompressibleConservativeVariables( const MeshPointer& meshPointer )
      : density( meshPointer ),
        momentum( meshPointer ),
        //pressure( meshPointer ),
        energy( meshPointer ){};
        
      void setMesh( const MeshPointer& meshPointer )
      {
         this->density->setMesh( meshPointer );
         this->momentum->setMesh( meshPointer );
         //this->pressure.setMesh( meshPointer );
         this->energy->setMesh( meshPointer );
      }
      
      template< typename Vector >
      void bind( const MeshPointer& meshPointer,
                 const Vector& data,
                 IndexType offset = 0 )
      {
         IndexType currentOffset( offset );
         this->density->bind( meshPointer, data, currentOffset );
         currentOffset += this->density->getDofs( meshPointer );
         for( IndexType i = 0; i < Dimensions; i++ )
         {
            ( *this->momentum )[ i ]->bind( meshPointer, data, currentOffset );
            currentOffset += ( *this->momentum )[ i ]->getDofs( meshPointer );
         }
         this->energy->bind( meshPointer, data, currentOffset );
      }
      
      IndexType getDofs( const MeshPointer& meshPointer ) const
      {
         return this->density->getDofs( meshPointer ) + 
            this->momentum->getDofs( meshPointer ) +
            this->energy->getDofs( meshPointer );
      }
      
      __cuda_callable__
      MeshFunctionPointer& getDensity()
      {
         return this->density;
      }

      __cuda_callable__
      const MeshFunctionPointer& getDensity() const
      {
         return this->density;
      }
      
      void setDensity( MeshFunctionPointer& density )
      {
         this->density = density;
      }
      
      __cuda_callable__
      MomentumFieldPointer& getMomentum()
      {
         return this->momentum;
      }
      
      __cuda_callable__
      const MomentumFieldPointer& getMomentum() const
      {
         return this->momentum;
      }
      
      void setMomentum( MomentumFieldPointer& momentum )
      {
         this->momentum = momentum;
      }
      
      /*
      __cuda_callable__
      MeshFunctionPointer& getPressure()
      {
         return this->pressure;
      }
      
      __cuda_callable__
      const MeshFunctionPointer& getPressure() const
      {
         return this->pressure;
      }
      
      void setPressure( MeshFunctionPointer& pressure )
      {
         this->pressure = pressure;
      }*/
      
      __cuda_callable__
      MeshFunctionPointer& getEnergy()
      {
         return this->energy;
      }
      
      __cuda_callable__
      const MeshFunctionPointer& getEnergy() const
      {
         return this->energy;
      }
      
      void setEnergy( MeshFunctionPointer& energy )
      {
         this->energy = energy;
      }
      
      void getVelocityField( VelocityFieldType& velocityField )
      {
         
      }

   protected:
      
      MeshFunctionPointer density;
      MomentumFieldPointer momentum;
      MeshFunctionPointer energy;
      
};

} // namespace TNL
