// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <array>

namespace TNL::Containers {

/**
 * \brief Directions for data synchronization in a distributed N-dimensional
 * array.
 *
 * It is treated as bitfield, i.e. the elementary enumerators represent
 * individual bits and compound enumerators are obtained by combining bits from
 * relevant elementary enumerators.
 *
 * \ingroup ndarray
 */
enum class SyncDirection : std::uint8_t
{
   All = 0xff,                                 //!< special value -- synchronize in all directions
   None = 0,                                   //!< special value -- no synchronization
   Right = 1 << 0,                             //!< synchronization from left to right (N >= 1, positive x-axis)
   Left = 1 << 1,                              //!< synchronization from right to left (N >= 1, negative x-axis)
   Top = 1 << 2,                               //!< synchronization from bottom to top (N >= 2, positive y-axis)
   Bottom = 1 << 3,                            //!< synchronization from top to bottom (N >= 2, negative y-axis)
   Front = 1 << 4,                             //!< synchronization from back to front (N >= 3, positive z-axis)
   Back = 1 << 5,                              //!< synchronization from front to back (N >= 3, negative z-axis)
   TopRight = Top | Right,                     //!< synchronization in the top-right direction
   TopLeft = Top | Left,                       //!< synchronization in the top-left direction
   BottomRight = Bottom | Right,               //!< synchronization in the bottom-right direction
   BottomLeft = Bottom | Left,                 //!< synchronization in the bottom-left direction
   BackRight = Back | Right,                   //!< synchronization in the back-right direction
   BackLeft = Back | Left,                     //!< synchronization in the back-left direction
   FrontRight = Front | Right,                 //!< synchronization in the front-right direction
   FrontLeft = Front | Left,                   //!< synchronization in the front-left direction
   BackTop = Back | Top,                       //!< synchronization in the back-top direction
   BackBottom = Back | Bottom,                 //!< synchronization in the back-bottom direction
   FrontTop = Front | Top,                     //!< synchronization in the front-top direction
   FrontBottom = Front | Bottom,               //!< synchronization in the front-bottom direction
   BackTopRight = Back | Top | Right,          //!< synchronization in the back-top-right direction
   BackTopLeft = Back | Top | Left,            //!< synchronization in the back-top-left direction
   BackBottomRight = Back | Bottom | Right,    //!< synchronization in the back-bottom-right direction
   BackBottomLeft = Back | Bottom | Left,      //!< synchronization in the back-bottom-left direction
   FrontTopRight = Front | Top | Right,        //!< synchronization in the front-top-right direction
   FrontTopLeft = Front | Top | Left,          //!< synchronization in the front-top-left direction
   FrontBottomRight = Front | Bottom | Right,  //!< synchronization in the front-bottom-right direction
   FrontBottomLeft = Front | Bottom | Left,    //!< synchronization in the front-bottom-left direction
};

/**
 * \brief Bitwise AND operator for \ref SyncDirection.
 *
 * \ingroup ndarray
 */
[[nodiscard]] inline SyncDirection
operator&( SyncDirection a, SyncDirection b )
{
   return static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) & static_cast< std::uint8_t >( b ) );
}

/**
 * \brief Bitwise OR operator for \ref SyncDirection.
 *
 * \ingroup ndarray
 */
[[nodiscard]] inline SyncDirection
operator|( SyncDirection a, SyncDirection b )
{
   return static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) | static_cast< std::uint8_t >( b ) );
}

/**
 * \brief Bitwise operator which clears all bits from `b` in `a`.
 *
 * This operator makes `a -= b` equivalent to `a &= ~b`, i.e. it clears all
 * bits from `b` in `a`.
 *
 * \returns reference to `a`
 *
 * \ingroup ndarray
 */
[[nodiscard]] inline SyncDirection&
operator-=( SyncDirection& a, SyncDirection b )
{
   a = static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) & ~static_cast< std::uint8_t >( b ) );
   return a;
}

/** \brief Namespace where synchronization patterns for distributed
 * N-dimensional arrays are defined.
 *
 * All names are inspired by the naming of velocity sets in LBM. Note that the
 * central velocity is never synchronized to a neighbor and thus
 * \ref SyncDirection::None is not included in any pattern.
 *
 * \ingroup ndarray
 */
namespace NDArraySyncPatterns {

// TODO: we can use constexpr std::vector since C++20
static constexpr std::array< SyncDirection, 2 > D1Q3 = { SyncDirection::Right, SyncDirection::Left };
static constexpr std::array< SyncDirection, 4 > D2Q5 = { SyncDirection::Right,
                                                         SyncDirection::Left,
                                                         SyncDirection::Top,
                                                         SyncDirection::Bottom };
static constexpr std::array< SyncDirection, 8 > D2Q9 = { SyncDirection::Right,       SyncDirection::Left,
                                                         SyncDirection::Top,         SyncDirection::Bottom,
                                                         SyncDirection::TopRight,    SyncDirection::TopLeft,
                                                         SyncDirection::BottomRight, SyncDirection::BottomLeft };
static constexpr std::array< SyncDirection, 6 > D3Q7 = { SyncDirection::Right,  SyncDirection::Left, SyncDirection::Top,
                                                         SyncDirection::Bottom, SyncDirection::Back, SyncDirection::Front };
static constexpr std::array< SyncDirection, 26 > D3Q27 = {
   SyncDirection::Right,
   SyncDirection::Left,
   SyncDirection::Top,
   SyncDirection::Bottom,
   SyncDirection::Back,
   SyncDirection::Front,
   SyncDirection::TopRight,
   SyncDirection::TopLeft,
   SyncDirection::BottomRight,
   SyncDirection::BottomLeft,
   SyncDirection::BackRight,
   SyncDirection::BackLeft,
   SyncDirection::FrontRight,
   SyncDirection::FrontLeft,
   SyncDirection::BackTop,
   SyncDirection::BackBottom,
   SyncDirection::FrontTop,
   SyncDirection::FrontBottom,
   SyncDirection::BackTopRight,
   SyncDirection::BackTopLeft,
   SyncDirection::BackBottomRight,
   SyncDirection::BackBottomLeft,
   SyncDirection::FrontTopRight,
   SyncDirection::FrontTopLeft,
   SyncDirection::FrontBottomRight,
   SyncDirection::FrontBottomLeft,
};

}  // namespace NDArraySyncPatterns

}  // namespace TNL::Containers
