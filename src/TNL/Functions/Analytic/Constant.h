// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL::Functions::Analytic {

template< int dimensions, typename Real = double >
class Constant : public Domain< dimensions, NonspaceDomain >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< dimensions, RealType >;

   __cuda_callable__
   Constant();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setConstant( const RealType& constant );

   [[nodiscard]] const RealType&
   getConstant() const;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   [[nodiscard]] __cuda_callable__
   inline RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   [[nodiscard]] __cuda_callable__
   inline RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const
   {
      return constant;
   }

   [[nodiscard]] __cuda_callable__
   inline RealType
   getValue( const Real& time = 0.0 ) const
   {
      return constant;
   }

protected:
   RealType constant;
};

template< int dimensions, typename Real >
std::ostream&
operator<<( std::ostream& str, const Constant< dimensions, Real >& f )
{
   str << "Constant function: constant = " << f.getConstant();
   return str;
}

}  // namespace TNL::Functions::Analytic

#include <TNL/Functions/Analytic/Constant_impl.h>
