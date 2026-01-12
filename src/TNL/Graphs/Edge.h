// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

/**
 * \brief Represents a weighted edge in a graph.
 *
 * The Edge structure stores information about a single edge in a graph, including
 * the source vertex, target vertex, and edge weight. It provides comparison operators
 * based on edge weights and accessor methods for all edge attributes.
 *
 * \tparam Real Type for the edge weight.
 * \tparam Index Type for vertex indices.
 */
template< typename Real = double, typename Index = int >
struct Edge
{
   using RealType = Real;
   using IndexType = Index;

   //! \brief Default constructor. Initializes source, target, and weight to zero.
   Edge() = default;

   /**
    * \brief Constructs an edge with specified source, target, and weight.
    *
    * \param source Index of the source vertex.
    * \param target Index of the target vertex.
    * \param weight Weight of the edge.
    */
   Edge( Index source, Index target, Real weight )
   : source( source ),
     target( target ),
     weight( weight )
   {}

   //! \brief Copy constructor.
   Edge( const Edge& ) = default;

   //! \brief Move constructor.
   Edge( Edge&& ) = default;

   //! \brief Copy assignment operator.
   Edge&
   operator=( const Edge& ) = default;

   //! \brief Move assignment operator.
   Edge&
   operator=( Edge&& ) = default;

   //! \brief Destructor.
   ~Edge() = default;

   /**
    * \brief Less-than comparison based on edge weight.
    * \param other The edge to compare with.
    * \return True if this edge's weight is less than the other edge's weight.
    */
   bool
   operator<( const Edge& other ) const
   {
      return weight < other.weight;
   }

   /**
    * \brief Greater-than comparison based on edge weight.
    * \param other The edge to compare with.
    * \return True if this edge's weight is greater than the other edge's weight.
    */
   bool
   operator>( const Edge& other ) const
   {
      return weight > other.weight;
   }

   /**
    * \brief Less-than-or-equal comparison based on edge weight.
    * \param other The edge to compare with.
    * \return True if this edge's weight is less than or equal to the other edge's weight.
    */
   bool
   operator<=( const Edge& other ) const
   {
      return weight <= other.weight;
   }

   /**
    * \brief Greater-than-or-equal comparison based on edge weight.
    * \param other The edge to compare with.
    * \return True if this edge's weight is greater than or equal to the other edge's weight.
    */
   bool
   operator>=( const Edge& other ) const
   {
      return weight >= other.weight;
   }

   /**
    * \brief Equality comparison.
    * \param other The edge to compare with.
    * \return True if both edges have the same source, target, and weight.
    */
   bool
   operator==( const Edge& other ) const
   {
      return source == other.source && target == other.target && weight == other.weight;
   }

   /**
    * \brief Inequality comparison.
    * \param other The edge to compare with.
    * \return True if edges differ in source, target, or weight.
    */
   bool
   operator!=( const Edge& other ) const
   {
      return ! ( *this == other );
   }

   /**
    * \brief Returns the source vertex index (const version).
    * \return Const reference to the source vertex index.
    */
   [[nodiscard]] const Index&
   getSource() const
   {
      return source;
   }

   /**
    * \brief Returns the source vertex index (modifiable version).
    * \return Reference to the source vertex index.
    */
   Index&
   getSource()
   {
      return source;
   }

   /**
    * \brief Returns the target vertex index (const version).
    * \return Const reference to the target vertex index.
    */
   [[nodiscard]] const Index&
   getTarget() const
   {
      return target;
   }

   /**
    * \brief Returns the target vertex index (modifiable version).
    * \return Reference to the target vertex index.
    */
   Index&
   getTarget()
   {
      return target;
   }

   /**
    * \brief Returns the edge weight (const version).
    * \return Const reference to the edge weight.
    */
   [[nodiscard]] const Real&
   getWeight() const
   {
      return weight;
   }

   /**
    * \brief Returns the edge weight (modifiable version).
    * \return Reference to the edge weight.
    */
   Real&
   getWeight()
   {
      return weight;
   }

protected:
   Index source = 0;   //!< Index of the source vertex.
   Index target = 0;   //!< Index of the target vertex.
   Real weight = 0.0;  //!< Weight of the edge.
};

/**
 * \brief Stream output operator for Edge.
 *
 * Outputs the edge in the format: (source, target, weight)
 *
 * \tparam Real Type for the edge weight.
 * \tparam Index Type for vertex indices.
 * \param os Output stream.
 * \param edge The edge to output.
 * \return Reference to the output stream.
 */
template< typename Real, typename Index >
std::ostream&
operator<<( std::ostream& os, const Edge< Real, Index >& edge )
{
   os << "(" << edge.getSource() << ", " << edge.getTarget() << ", " << edge.getWeight() << ")";
   return os;
}

}  // namespace TNL::Graphs
