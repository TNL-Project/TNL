// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL::Functions::Analytic {

template< int dimensions, typename Real = double >
class ParaboloidBase : public Functions::Domain< dimensions, SpaceDomain >
{
public:
   ParaboloidBase();

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setXCenter( const Real& waveLength );

   [[nodiscard]] Real
   getXCenter() const;

   void
   setYCenter( const Real& waveLength );

   [[nodiscard]] Real
   getYCenter() const;

   void
   setZCenter( const Real& waveLength );

   [[nodiscard]] Real
   getZCenter() const;

   void
   setCoefficient( const Real& coefficient );

   [[nodiscard]] Real
   getCoefficient() const;

   void
   setOffset( const Real& offset );

   [[nodiscard]] Real
   getOffset() const;

protected:
   Real xCenter, yCenter, zCenter, coefficient, radius;
};

template< int Dimensions, typename Real >
class Paraboloid
{};

template< typename Real >
class Paraboloid< 1, Real > : public ParaboloidBase< 1, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 1, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   [[nodiscard]] __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   [[nodiscard]] __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const
   {
      return this->getPartialDerivative< 0, 0, 0 >( v, time );
   }
};

template< typename Real >
class Paraboloid< 2, Real > : public ParaboloidBase< 2, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 2, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   [[nodiscard]] __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   [[nodiscard]] __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const
   {
      return this->getPartialDerivative< 0, 0, 0 >( v, time );
   }
};

template< typename Real >
class Paraboloid< 3, Real > : public ParaboloidBase< 3, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 3, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   [[nodiscard]] __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   [[nodiscard]] __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const
   {
      return this->getPartialDerivative< 0, 0, 0 >( v, time );
   }
};

template< int Dimensions, typename Real >
std::ostream&
operator<<( std::ostream& str, const Paraboloid< Dimensions, Real >& f )
{
   str << "SDF Paraboloid function: amplitude = " << f.getCoefficient() << " offset = " << f.getOffset();
   return str;
}

}  // namespace TNL::Functions::Analytic

#include <TNL/Functions/Analytic/Paraboloid_impl.h>
