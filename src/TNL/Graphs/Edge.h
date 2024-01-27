// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

template< typename Real = double, typename Index = int >
struct Edge
{
   using RealType = Real;
   using IndexType = Index;

   Edge() = default;

   Edge( Index source, Index target, Real weight ) : source( source ), target( target ), weight( weight ) {}

   Edge( const Edge& ) = default;

   Edge( Edge&& ) = default;

   Edge&
   operator=( const Edge& ) = default;

   Edge&
   operator=( Edge&& ) = default;

   ~Edge() = default;

   bool
   operator<( const Edge& other ) const
   {
      return weight < other.weight;
   }

   bool
   operator>( const Edge& other ) const
   {
      return weight > other.weight;
   }

   bool
   operator<=( const Edge& other ) const
   {
      return weight <= other.weight;
   }

   bool
   operator>=( const Edge& other ) const
   {
      return weight >= other.weight;
   }

   bool
   operator==( const Edge& other ) const
   {
      return source == other.source && target == other.target && weight == other.weight;
   }

   bool
   operator!=( const Edge& other ) const
   {
      return ! ( *this == other );
   }

   [[nodiscard]] const Index&
   getSource() const
   {
      return source;
   }

   Index&
   getSource()
   {
      return source;
   }

   [[nodiscard]] const Index&
   getTarget() const
   {
      return target;
   }

   Index&
   getTarget()
   {
      return target;
   }

   [[nodiscard]] const Real&
   getWeight() const
   {
      return weight;
   }

   Real&
   getWeight()
   {
      return weight;
   }

protected:
   Index source = 0;
   Index target = 0;
   Real weight = 0.0;
};

template< typename Real, typename Index >
std::ostream&
operator<<( std::ostream& os, const Edge< Real, Index >& edge )
{
   os << "(" << edge.getSource() << ", " << edge.getTarget() << ", " << edge.getWeight() << ")";
   return os;
}

}  // namespace TNL::Graphs
