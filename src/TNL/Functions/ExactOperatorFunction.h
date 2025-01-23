// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functions/Domain.h>

namespace TNL::Functions {

template< typename Operator, typename Function >
class ExactOperatorFunction : public Domain< Operator::getDomainDimension(), SpaceDomain >
{
   static_assert( Operator::getDomainDimension() == Function::getDomainDimension(),
                  "Operator and function have different number of domain dimensions." );

public:
   typedef Operator OperatorType;
   typedef Function FunctionType;
   typedef typename FunctionType::RealType RealType;
   typedef typename FunctionType::PointType PointType;

   [[nodiscard]] static constexpr int
   getDomainDimension()
   {
      return Operator::getDomainDimension();
   }

   ExactOperatorFunction( const OperatorType& operator_, const FunctionType& function )
   : operator_( operator_ ),
     function( function )
   {}

   [[nodiscard]] __cuda_callable__
   RealType
   operator()( const PointType& vertex, const RealType& time ) const
   {
      return this->operator_( function, vertex, time );
   }

protected:
   const OperatorType& operator_;

   const FunctionType& function;
};

}  // namespace TNL::Functions
