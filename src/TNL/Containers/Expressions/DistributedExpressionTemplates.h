// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Expressions/DistributedVerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Distributed unary expression template
template< typename T1, typename Operation >
struct DistributedUnaryExpressionTemplate;

template< typename T1, typename Operation >
struct HasEnabledDistributedExpressionTemplates< DistributedUnaryExpressionTemplate< T1, Operation > > : std::true_type
{};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct DistributedBinaryExpressionTemplate
{};

template< typename T1, typename T2, typename Operation, ExpressionVariableType T1Type, ExpressionVariableType T2Type >
struct HasEnabledDistributedExpressionTemplates< DistributedBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType =
      BinaryExpressionTemplate< typename T1::ConstLocalViewType, typename T2::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the left operand." );
   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the right operand." );
   static_assert( std::is_same_v< typename T1::DeviceType, typename T2::DeviceType >,
                  "Attempt to mix operands which have different DeviceType." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ),
     op2( b )
   {
      if( op1.getSize() != op2.getSize() )
         throw std::logic_error( "Attempt to mix operands with different sizes." );
      if( op1.getLocalRange() != op2.getLocalRange() )
         throw std::logic_error( "Distributed expressions are supported only on vectors which are distributed the same way." );
      if( op1.getGhosts() != op2.getGhosts() )
         throw std::logic_error( "Distributed expressions are supported only on vectors which are distributed the same way." );
      if( op1.getCommunicator() != op2.getCommunicator() )
         throw std::logic_error( "Distributed expressions are supported only on vectors within the same communicator." );
   }

   [[nodiscard]] RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   [[nodiscard]] IndexType
   getSize() const
   {
      return op1.getSize();
   }

   [[nodiscard]] LocalRangeType
   getLocalRange() const
   {
      return op1.getLocalRange();
   }

   [[nodiscard]] IndexType
   getGhosts() const
   {
      return op1.getGhosts();
   }

   [[nodiscard]] const MPI::Comm&
   getCommunicator() const
   {
      return op1.getCommunicator();
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2.getConstLocalView() );
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2.getConstLocalViewWithGhosts() );
   }

   [[nodiscard]] std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   [[nodiscard]] int
   getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op1.waitForSynchronization();
      op2.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >() ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType, T2, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the left operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ),
     op2( b )
   {}

   [[nodiscard]] RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   [[nodiscard]] IndexType
   getSize() const
   {
      return op1.getSize();
   }

   [[nodiscard]] LocalRangeType
   getLocalRange() const
   {
      return op1.getLocalRange();
   }

   [[nodiscard]] IndexType
   getGhosts() const
   {
      return op1.getGhosts();
   }

   [[nodiscard]] const MPI::Comm&
   getCommunicator() const
   {
      return op1.getCommunicator();
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2 );
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2 );
   }

   [[nodiscard]] std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   [[nodiscard]] int
   getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op1.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2 op2;
};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >(), std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using LocalRangeType = typename T2::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< T1, typename T2::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T2::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the right operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ),
     op2( b )
   {}

   [[nodiscard]] RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   [[nodiscard]] IndexType
   getSize() const
   {
      return op2.getSize();
   }

   [[nodiscard]] LocalRangeType
   getLocalRange() const
   {
      return op2.getLocalRange();
   }

   [[nodiscard]] IndexType
   getGhosts() const
   {
      return op2.getGhosts();
   }

   [[nodiscard]] const MPI::Comm&
   getCommunicator() const
   {
      return op2.getCommunicator();
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalView() );
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalViewWithGhosts() );
   }

   [[nodiscard]] std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op2.getSynchronizer();
   }

   [[nodiscard]] int
   getValuesPerElement() const
   {
      return op2.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op2.waitForSynchronization();
   }

protected:
   const T1 op1;
   const T2& op2;
};

////
// Distributed unary expression template
template< typename T1, typename Operation >
struct DistributedUnaryExpressionTemplate
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = UnaryExpressionTemplate< typename T1::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed unary expression templates - distributed expression templates are not "
                  "enabled for the operand." );

   // the constructor is explicit to prevent issues with the ternary operator,
   // see https://gitlab.com/tnl-project/tnl/-/issues/140
   explicit DistributedUnaryExpressionTemplate( const T1& a )
   : operand( a )
   {}

   [[nodiscard]] RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   [[nodiscard]] IndexType
   getSize() const
   {
      return operand.getSize();
   }

   [[nodiscard]] LocalRangeType
   getLocalRange() const
   {
      return operand.getLocalRange();
   }

   [[nodiscard]] IndexType
   getGhosts() const
   {
      return operand.getGhosts();
   }

   [[nodiscard]] const MPI::Comm&
   getCommunicator() const
   {
      return operand.getCommunicator();
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( operand.getConstLocalView() );
   }

   [[nodiscard]] ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( operand.getConstLocalViewWithGhosts() );
   }

   [[nodiscard]] std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return operand.getSynchronizer();
   }

   [[nodiscard]] int
   getValuesPerElement() const
   {
      return operand.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      operand.waitForSynchronization();
   }

protected:
   const T1& operand;
};

#ifndef DOXYGEN_ONLY

   #define TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( fname, functor )                                    \
      template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true > \
      auto fname( const ET1& a )                                                                      \
      {                                                                                               \
         return DistributedUnaryExpressionTemplate< ET1, functor >( a );                              \
      }

   #define TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( fname, functor )                                                       \
      template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true > \
      auto fname( const ET1& a, const ET2& b )                                                                            \
      {                                                                                                                   \
         return DistributedBinaryExpressionTemplate< ET1, ET2, functor >( a, b );                                         \
      }

// NOTE: The list of functions and operators defined for distributed vectors
// should be kept in sync with the list of functions and operators defined for
// (normal) vectors - see ExpressionTemplates.h.
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator+, TNL::Plus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator-, TNL::Minus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator*, TNL::Multiplies )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator/, TNL::Divides )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator%, TNL::Modulus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator&&, TNL::LogicalAnd )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator||, TNL::LogicalOr )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator&, TNL::BitAnd )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator|, TNL::BitOr )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator^, TNL::BitXor )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( equalTo, TNL::EqualTo )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( notEqualTo, TNL::NotEqualTo )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( greater, TNL::Greater )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( less, TNL::Less )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( greaterEqual, TNL::GreaterEqual )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( lessEqual, TNL::LessEqual )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( minimum, TNL::Min )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( maximum, TNL::Max )

TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator+, TNL::UnaryPlus )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator-, TNL::UnaryMinus )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator!, TNL::LogicalNot )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator~, TNL::BitNot )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( abs, TNL::Abs )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( exp, TNL::Exp )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sqr, TNL::Sqr )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sqrt, TNL::Sqrt )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cbrt, TNL::Cbrt )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log, TNL::Log )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log10, TNL::Log10 )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log2, TNL::Log2 )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sin, TNL::Sin )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cos, TNL::Cos )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( tan, TNL::Tan )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( asin, TNL::Asin )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( acos, TNL::Acos )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( atan, TNL::Atan )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sinh, TNL::Sinh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cosh, TNL::Cosh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( tanh, TNL::Tanh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( asinh, TNL::Asinh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( acosh, TNL::Acosh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( atanh, TNL::Atanh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( floor, TNL::Floor )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( ceil, TNL::Ceil )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sign, TNL::Sign )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( conj, TNL::Conj )

   #undef TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION
   #undef TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION

////
// Pow
template< typename ET1, typename Real, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
pow( const ET1& a, const Real& exp )
{
   return DistributedBinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Cast
template< typename ResultType, typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
cast( const ET1& a )
{
   using CastOperation = typename Cast< ResultType >::Operation;
   return DistributedUnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
auto
operator,( const ET1& a, const ET2& b )
{
   if constexpr( is_complex_v< typename ET1::ValueType > ) {
      return sum( conj( a ) * b );
   }
   else {
      return sum( a * b );
   }
}

template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
auto
dot( const ET1& a, const ET2& b )
{
   return a, b;
}

////
// Vertical operations
template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
min( const ET1& a )
{
   return DistributedExpressionMin( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
argMin( const ET1& a )
{
   return DistributedExpressionArgMin( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
max( const ET1& a )
{
   return DistributedExpressionMax( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
argMax( const ET1& a )
{
   return DistributedExpressionArgMax( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
sum( const ET1& a )
{
   return DistributedExpressionSum( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( sqr( a ) ) );
}

template< typename ET1, typename Real, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
lpNorm( const ET1& a, const Real& p )
   // since (1.0 / p) has type double, TNL::pow returns double
   -> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   using TNL::pow;
   return pow( sum( pow( abs( a ), p ) ), 1.0 / p );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
product( const ET1& a )
{
   return DistributedExpressionProduct( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
all( const ET1& a )
{
   return DistributedExpressionAll( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
any( const ET1& a )
{
   return DistributedExpressionAny( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
argAny( const ET1& a )
{
   return DistributedExpressionArgAny( a );
}

////
// Comparison operator ==
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator==( const ET1& a, const ET2& b )
{
   MPI_Comm communicator = MPI_COMM_NULL;

   // we can't return all( equalTo( a, b ) ) because we want to allow comparison on different devices and
   // DistributedBinaryExpressionTemplate does not allow that
   bool localResult = false;
   if constexpr( getExpressionVariableType< ET1, ET2 >() == VectorExpressionVariable
                 && getExpressionVariableType< ET2, ET1 >() == VectorExpressionVariable )
   {
      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      communicator = a.getCommunicator();
      localResult = a.getLocalRange() == b.getLocalRange() && a.getGhosts() == b.getGhosts() && a.getSize() == b.getSize() &&
                    // compare without ghosts
                    a.getConstLocalView() == b.getConstLocalView();
   }
   else if constexpr( getExpressionVariableType< ET1, ET2 >() == VectorExpressionVariable ) {
      communicator = a.getCommunicator();
      localResult = a.getConstLocalView() == b;
   }
   else if constexpr( getExpressionVariableType< ET2, ET1 >() == VectorExpressionVariable ) {
      communicator = b.getCommunicator();
      localResult = a == b.getConstLocalView();
   }
   bool result = true;
   if( communicator != MPI_COMM_NULL )
      MPI::Allreduce( &localResult, &result, 1, MPI_LAND, communicator );
   return result;
}

////
// Comparison operator !=
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator!=( const ET1& a, const ET2& b )
{
   return ! operator==( a, b );
}

////
// Lexicographical comparison operators
template< typename ET1,
          typename ET2,
          typename...,
          std::enable_if_t< HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
                               && HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value,
                            bool > = true >
constexpr bool
operator<( const ET1& a, const ET2& b )
{
   // TODO: The use of `argAny` might not be the most efficient. It might be
   // better to implement some function like `findFirst` for this purpose.
   auto [ notEqual, idx ] = argAny( notEqualTo( a, b ) );
   if( notEqual ) {
      auto range = a.getLocalRange();
      bool localResult = false;
      if( idx >= range.getBegin() && idx < range.getEnd() )
         localResult = ( a.getElement( idx ) < b.getElement( idx ) );
      bool result = false;
      MPI::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicator() );
      return result;
   }
   return false;
}

template< typename ET1,
          typename ET2,
          typename...,
          std::enable_if_t< HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
                               && HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value,
                            bool > = true >
constexpr bool
operator<=( const ET1& a, const ET2& b )
{
   return ! operator>( a, b );
}

template< typename ET1,
          typename ET2,
          typename...,
          std::enable_if_t< HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
                               && HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value,
                            bool > = true >
constexpr bool
operator>( const ET1& a, const ET2& b )
{
   // TODO: The use of `argAny` might not be the most efficient. It might be
   // better to implement some function like `findFirst` for this purpose.
   auto [ notEqual, idx ] = argAny( notEqualTo( a, b ) );
   if( notEqual ) {
      auto range = a.getLocalRange();
      bool localResult = false;
      if( idx >= range.getBegin() && idx < range.getEnd() )
         localResult = ( a.getElement( idx ) > b.getElement( idx ) );
      bool result = false;
      MPI::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicator() );
      return result;
   }
   return false;
}

template< typename ET1,
          typename ET2,
          typename...,
          std::enable_if_t< HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
                               && HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value,
                            bool > = true >
constexpr bool
operator>=( const ET1& a, const ET2& b )
{
   return ! operator<( a, b );
}

////
// Output stream
template< typename T1, typename T2, typename Operation >
std::ostream&
operator<<( std::ostream& str, const DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

template< typename T, typename Operation >
std::ostream&
operator<<( std::ostream& str, const DistributedUnaryExpressionTemplate< T, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

#endif  // DOXYGEN_ONLY

}  // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for DistributedVector and DistributedVectorView
using Expressions::operator!;
using Expressions::operator~;
using Expressions::operator+;
using Expressions::operator-;
using Expressions::operator*;
using Expressions::operator/;
using Expressions::operator%;
using Expressions::operator&&;
using Expressions::operator||;
using Expressions::operator&;
using Expressions::operator|;
using Expressions::operator^;
using Expressions::operator, ;
using Expressions::operator==;
using Expressions::operator!=;
using Expressions::operator<;
using Expressions::operator<=;
using Expressions::operator>;
using Expressions::operator>=;

using Expressions::equalTo;
using Expressions::greater;
using Expressions::greaterEqual;
using Expressions::less;
using Expressions::lessEqual;
using Expressions::notEqualTo;

// Make all functions visible in the TNL::Containers namespace
using Expressions::abs;
using Expressions::acos;
using Expressions::acosh;
using Expressions::all;
using Expressions::any;
using Expressions::argAny;
using Expressions::argMax;
using Expressions::argMin;
using Expressions::asin;
using Expressions::asinh;
using Expressions::atan;
using Expressions::atanh;
using Expressions::cast;
using Expressions::cbrt;
using Expressions::ceil;
using Expressions::conj;
using Expressions::cos;
using Expressions::cosh;
using Expressions::dot;
using Expressions::exp;
using Expressions::floor;
using Expressions::l1Norm;
using Expressions::l2Norm;
using Expressions::log;
using Expressions::log10;
using Expressions::log2;
using Expressions::lpNorm;
using Expressions::max;
using Expressions::maximum;
using Expressions::maxNorm;
using Expressions::min;
using Expressions::minimum;
using Expressions::pow;
using Expressions::product;
using Expressions::sign;
using Expressions::sin;
using Expressions::sinh;
using Expressions::sqr;
using Expressions::sqrt;
using Expressions::sum;
using Expressions::tan;
using Expressions::tanh;

}  // namespace Containers

// Make all functions visible in the main TNL namespace
using Containers::abs;
using Containers::acos;
using Containers::acosh;
using Containers::all;
using Containers::any;
using Containers::argAny;
using Containers::argMax;
using Containers::argMin;
using Containers::asin;
using Containers::asinh;
using Containers::atan;
using Containers::atanh;
using Containers::cast;
using Containers::cbrt;
using Containers::ceil;
using Containers::conj;
using Containers::cos;
using Containers::cosh;
using Containers::dot;
using Containers::equalTo;
using Containers::exp;
using Containers::floor;
using Containers::greater;
using Containers::greaterEqual;
using Containers::l1Norm;
using Containers::l2Norm;
using Containers::less;
using Containers::lessEqual;
using Containers::log;
using Containers::log10;
using Containers::log2;
using Containers::lpNorm;
using Containers::max;
using Containers::maximum;
using Containers::maxNorm;
using Containers::min;
using Containers::minimum;
using Containers::notEqualTo;
using Containers::pow;
using Containers::product;
using Containers::sign;
using Containers::sin;
using Containers::sinh;
using Containers::sqr;
using Containers::sqrt;
using Containers::sum;
using Containers::tan;
using Containers::tanh;

////
// Evaluation with reduction
template< typename Vector,
          typename ET1,
          typename Reduction,
          typename Result,
          std::enable_if_t< Containers::Expressions::HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value,
                            bool > = true >
Result
evaluateAndReduce( Vector& lhs, const ET1& expression, const Reduction& reduction, const Result& zero )
{
   using IndexType = typename Vector::IndexType;

   Result result = zero;
   const MPI::Comm& communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local result
      auto local_lhs = lhs.getConstLocalView();
      Result localResult = evaluateAndReduce( local_lhs, expression.getConstLocalView(), reduction, zero );

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< Result[] > dataForScatter{ new Result[ nproc ] };
      for( int i = 0; i < nproc; i++ )
         dataForScatter[ i ] = localResult;
      std::unique_ptr< Result[] > gatheredResults{ new Result[ nproc ] };
      // NOTE: exchanging general data types does not work with MPI
      // MPI::Alltoall( dataForScatter.get(), 1, gatheredResults.get(), 1, communicator );
      MPI::Alltoall(
         (char*) dataForScatter.get(), sizeof( Result ), (char*) gatheredResults.get(), sizeof( Result ), communicator );

      // compute the global reduction over MPI ranks
      auto fetch = [ &gatheredResults ]( IndexType i )
      {
         return gatheredResults[ i ];
      };
      result = Algorithms::reduce< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, reduction, zero );
   }
   return result;
}

////
// Addition and reduction
template< typename Vector,
          typename ET1,
          typename Reduction,
          typename Result,
          std::enable_if_t< Containers::Expressions::HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value,
                            bool > = true >
Result
addAndReduce( Vector& lhs, const ET1& expression, const Reduction& reduction, const Result& zero )
{
   using IndexType = typename Vector::IndexType;

   Result result = zero;
   const MPI::Comm& communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local result
      auto local_lhs = lhs.getConstLocalView();
      Result localResult = addAndReduce( local_lhs, expression.getConstLocalView(), reduction, zero );

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< Result[] > dataForScatter{ new Result[ nproc ] };
      for( int i = 0; i < nproc; i++ )
         dataForScatter[ i ] = localResult;
      std::unique_ptr< Result[] > gatheredResults{ new Result[ nproc ] };
      // NOTE: exchanging general data types does not work with MPI
      // MPI::Alltoall( dataForScatter.get(), 1, gatheredResults.get(), 1, communicator );
      MPI::Alltoall(
         (char*) dataForScatter.get(), sizeof( Result ), (char*) gatheredResults.get(), sizeof( Result ), communicator );

      // compute the global reduction over MPI ranks
      auto fetch = [ &gatheredResults ]( IndexType i )
      {
         return gatheredResults[ i ];
      };
      result = Algorithms::reduce< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, reduction, zero );
   }
   return result;
}

////
// Addition and reduction
template< typename Vector,
          typename ET1,
          typename Reduction,
          typename Result,
          std::enable_if_t< Containers::Expressions::HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value,
                            bool > = true >
Result
addAndReduceAbs( Vector& lhs, const ET1& expression, const Reduction& reduction, const Result& zero )
{
   using IndexType = typename Vector::IndexType;

   Result result = zero;
   const MPI::Comm& communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local result
      auto local_lhs = lhs.getLocalView();
      Result localResult = addAndReduceAbs( local_lhs, expression.getConstLocalView(), reduction, zero );

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< Result[] > dataForScatter{ new Result[ nproc ] };
      for( int i = 0; i < nproc; i++ )
         dataForScatter[ i ] = localResult;
      std::unique_ptr< Result[] > gatheredResults{ new Result[ nproc ] };
      // NOTE: exchanging general data types does not work with MPI
      // MPI::Alltoall( dataForScatter.get(), 1, gatheredResults.get(), 1, communicator );
      MPI::Alltoall(
         (char*) dataForScatter.get(), sizeof( Result ), (char*) gatheredResults.get(), sizeof( Result ), communicator );

      // compute the global reduction over MPI ranks
      auto fetch = [ &gatheredResults ]( IndexType i )
      {
         return gatheredResults[ i ];
      };
      result = Algorithms::reduce< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, reduction, zero );
   }
   return result;
}

}  // namespace TNL

// Helper TNL_ASSERT_ALL_* macros
#include "Assert.h"
