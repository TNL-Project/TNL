#pragma once

#include <array>
#include <cstddef>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Math.h>

template< typename Value = double >
struct Fehlberg2
{
   using ValueType = Value;

   //! [Stages]
   static constexpr size_t Stages = 3;
   //! [Stages]

   static constexpr size_t
   getStages()
   {
      return Stages;
   }

   //! [Adaptivity]
   static constexpr bool
   isAdaptive()
   {
      return true;
   }
   //! [Adaptivity]

   static constexpr ValueType
   getCoefficient( const size_t stage, const size_t i )
   {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( size_t i )
   {
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( size_t i )
   {
      return higher_order_update_coefficients[ i ];
   }

   //! [Error coefficients]
   static constexpr ValueType
   getErrorCoefficient( size_t i )
   {
      return higher_order_update_coefficients[ i ] - lower_order_update_coefficients[ i ];
   }
   //! [Error coefficients]

protected:
   // clang-format off
   //! [k coefficients definition]
   static constexpr std::array< std::array< Value, Stages >, Stages > k_coefficients{
      std::array< Value, Stages >{ 0.0,       0.0,         0.0 },
      std::array< Value, Stages >{ 1.0/2.0,   0.0,         0.0 },
      std::array< Value, Stages >{ 1.0/256.0, 255.0/256.0, 0.0 }
   };
   //! [k coefficients definition]

   //! [Time coefficients definition]
   static constexpr std::array< Value, Stages > time_coefficients{ 0.0, 1.0/2.0, 1.0 };
   //! [Time coefficients definition]

   //! [Update coefficients definition]
   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 1.0/512.0, 255.0/256.0, 1.0/512.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 1.0/256.0, 255.0/256.0, 0.0 };
   //! [Update coefficients definition]
   // clang-format on
};
