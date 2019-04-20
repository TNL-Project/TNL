/***************************************************************************
                          CompressibleConservativeVariables.h  -  description
                             -------------------
    begin                : Feb 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {
   
template< typename Mesh >
class PhysicalVariablesGetter
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef CompressibleConservativeVariables< MeshType > ConservativeVariablesType;
      typedef Pointers::SharedPointer< ConservativeVariablesType > ConservativeVariablesPointer;
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< VelocityFieldType > VelocityFieldPointer;
      
      class VelocityGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            VelocityGetter( MeshFunctionPointer density, 
                            MeshFunctionPointer momentum )
            : density( density ), momentum( momentum ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                        const RealType& time = 0.0 ) const
            {
               if( density.template getData< DeviceType >()( meshEntity ) == 0.0 )
                  return 0;
               else
                  return momentum.template getData< DeviceType >()( meshEntity ) / 
                         density.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer density, momentum;
      };
      
      class PressureGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            PressureGetter( MeshFunctionPointer density,
                            MeshFunctionPointer energy, 
                            VelocityFieldPointer momentum,
                            const RealType& gamma )
            : density( density ), energy( energy ), momentum( momentum ), gamma( gamma ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                 const RealType& time = 0.0 ) const
            {
               const RealType e = energy.template getData< DeviceType >()( meshEntity );
               const RealType rho = density.template getData< DeviceType >()( meshEntity );
               const RealType momentumNorm = momentum.template getData< DeviceType >().getVector( meshEntity ).lpNorm( 2.0 );
               if( rho == 0.0 )
                  return 0;
               else
                  return ( gamma - 1.0 ) * ( e - 0.5 * momentumNorm * momentumNorm / rho );
            }
            
         protected:
            const MeshFunctionPointer density, energy;
            const VelocityFieldPointer momentum;
            const RealType gamma;
      };

      class TurbulentEnergyGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            TurbulentEnergyGetter( MeshFunctionPointer density,
                                   MeshFunctionPointer turbulentEnergyXDensity )
            : density( density ), turbulentEnergyXDensity( turbulentEnergyXDensity ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                 const RealType& time = 0.0 ) const
            {
               if( density.template getData< DeviceType >()( meshEntity ) == 0.0 )
                  return 0;
               else
                  return turbulentEnergyXDensity.template getData< DeviceType >()( meshEntity ) / 
                         density.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer density, turbulentEnergyXDensity;
      };

      class DisipationGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            DisipationGetter( MeshFunctionPointer density,
                              MeshFunctionPointer disipationXDensity )
            : density( density ), disipationXDensity( disipationXDensity ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                 const RealType& time = 0.0 ) const
            {
               if( density.template getData< DeviceType >()( meshEntity ) == 0.0 )
                  return 0;
               else
                  return disipationXDensity.template getData< DeviceType >()( meshEntity ) / 
                         density.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer density, disipationXDensity;
      };

      class TurbulentViscosityGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            TurbulentViscosityGetter( MeshFunctionPointer turbulentEnergy,
                                      MeshFunctionPointer disipation )
            :turbulentEnergy( turbulentEnergy ), disipation( disipation ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                 const RealType& time = 0.0 ) const
            {
               if( disipation.template getData< DeviceType >()( meshEntity ) == 0.0 )
                  return 0;
               else
                  return turbulentEnergy.template getData< DeviceType >()( meshEntity )
                       / disipation.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer disipation, turbulentEnergy;
      };      

      
      void getVelocity( const ConservativeVariablesPointer& conservativeVariables,
                        VelocityFieldPointer& velocity )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, VelocityGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            Pointers::SharedPointer< VelocityGetter, DeviceType > velocityGetter( conservativeVariables->getDensity(),
                                                                        ( *conservativeVariables->getMomentum() )[ i ] );
            evaluator.evaluate( ( *velocity )[ i ], velocityGetter );
         }
      }
      
      void getPressure( const ConservativeVariablesPointer& conservativeVariables,
                        const RealType& gamma,
                        MeshFunctionPointer& pressure )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, PressureGetter > evaluator;
         Pointers::SharedPointer< PressureGetter, DeviceType > pressureGetter( conservativeVariables->getDensity(),
                                                                     conservativeVariables->getEnergy(),
                                                                     conservativeVariables->getMomentum(),
                                                                     gamma );
         evaluator.evaluate( pressure, pressureGetter );
      }

      void getTurbulentEnergy( const ConservativeVariablesPointer& conservativeVariables,
                               MeshFunctionPointer& turbulentEnergy_no_rho )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, TurbulentEnergyGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            Pointers::SharedPointer< TurbulentEnergyGetter, DeviceType > turbulentEnergyGetter( conservativeVariables->getDensity(),
                                                                                                conservativeVariables->getTurbulentEnergy() );
            evaluator.evaluate( turbulentEnergy_no_rho, turbulentEnergyGetter );
         }
      }

      void getDisipation( const ConservativeVariablesPointer& conservativeVariables,
                                MeshFunctionPointer& disipation_no_rho )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, DisipationGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            Pointers::SharedPointer< DisipationGetter, DeviceType > disipationGetter( conservativeVariables->getDensity(),
                                                                                           conservativeVariables->getDisipation() );
            evaluator.evaluate( disipation_no_rho, disipationGetter );
         }
      }

      void getTurbulentViscosity(       MeshFunctionPointer& turbulentEnergy_no_rho,
                                        MeshFunctionPointer& disipation_no_rho,
                                        MeshFunctionPointer& turbulentViscosity )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, TurbulentViscosityGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            Pointers::SharedPointer< TurbulentViscosityGetter, DeviceType > turbulentViscosityGetter( turbulentEnergy_no_rho,
                                                                                                      disipation_no_rho);
            evaluator.evaluate( turbulentViscosity, turbulentViscosityGetter );
         }
      }
      
};
   
} //namespace TNL
