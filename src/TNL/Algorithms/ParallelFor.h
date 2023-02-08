// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/ParallelFor1D.h>
#include <TNL/Algorithms/detail/ParallelFor2D.h>
#include <TNL/Algorithms/detail/ParallelFor3D.h>

/****
 * The implementation of ParallelFor is not meant to provide maximum performance
 * at every cost, but maximum flexibility for operating with data stored on the
 * device.
 *
 * The grid-stride loop for CUDA has been inspired by Nvidia's blog post:
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 *
 * Implemented by: Jakub Klinkovsky
 */

namespace TNL {
/**
 * \brief Namespace for fundamental TNL algorithms
 *
 * It contains algorithms like for-loops, memory operations, (parallel) reduction,
 * multireduction, scan etc.
 */
namespace Algorithms {

/**
 * \brief Parallel for loop for one dimensional interval of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref TNL::Devices::Host, \ref TNL::Devices::Cuda or
 *    \ref TNL::Devices::Sequential.
 */
template< typename Device = Devices::Sequential >
struct ParallelFor
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param start is the left bound of the iteration range `[begin, end)`.
    * \param end is the right bound of the iteration range `[begin, end)`.
    * \param f is the function to be called in each iteration.
    * \param args are additional parameters to be passed to the function f.
    *
    * \par Example
    * \include Algorithms/ParallelForExample.cpp
    * \par Output
    * \include ParallelForExample.out
    *
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index start, Index end, Function f, FunctionArgs... args )
   {
      typename Device::LaunchConfiguration launch_config;
      exec( start, end, launch_config, f, args... );
   }

   /**
    * \brief Overload with custom launch configuration (which is ignored for
    * \ref TNL::Devices::Sequential).
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index start, Index end, typename Device::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
   {
      detail::ParallelFor1D< Device >::exec( start, end, launch_config, f, args... );
   }
};

/**
 * \brief Parallel for loop for two dimensional domain of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref TNL::Devices::Host, \ref TNL::Devices::Cuda or
 *    \ref TNL::Devices::Sequential.
 */
template< typename Device = Devices::Sequential >
struct ParallelFor2D
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param startX the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param startY the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param endX the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param endY the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param f is the function to be called in each iteration
    * \param args are additional parameters to be passed to the function f.
    *
    * The function f is called for each iteration as
    *
    * \code
    * f( i, j, args... )
    * \endcode
    *
    * where the first parameter is changing more often than the second one.
    *
    * \par Example
    * \include Algorithms/ParallelForExample-2D.cpp
    * \par Output
    * \include ParallelForExample-2D.out
    *
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
      typename Device::LaunchConfiguration launch_config;
      exec( startX, startY, endX, endY, launch_config, f, args... );
   }

   /**
    * \brief Overload with custom launch configuration (which is ignored for
    * \ref TNL::Devices::Sequential).
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index endX,
         Index endY,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      detail::ParallelFor2D< Device >::exec( startX, startY, endX, endY, launch_config, f, args... );
   }
};

/**
 * \brief Parallel for loop for three dimensional domain of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref TNL::Devices::Host, \ref TNL::Devices::Cuda or
 *    \ref TNL::Devices::Sequential.
 */
template< typename Device = Devices::Sequential >
struct ParallelFor3D
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param startX the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param startY the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param startZ the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endX the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endY the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endZ the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param f is the function to be called in each iteration
    * \param args are additional parameters to be passed to the function f.
    *
    * The function f is called for each iteration as
    *
    * \code
    * f( i, j, k, args... )
    * \endcode
    *
    * where the first parameter is changing the most often.
    *
    * \par Example
    * \include Algorithms/ParallelForExample-3D.cpp
    * \par Output
    * \include ParallelForExample-3D.out
    *
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
      typename Device::LaunchConfiguration launch_config;
      exec( startX, startY, startZ, endX, endY, endZ, launch_config, f, args... );
   }

   /**
    * \brief Overload with custom launch configuration (which is ignored for
    * \ref TNL::Devices::Sequential).
    */
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index startZ,
         Index endX,
         Index endY,
         Index endZ,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      detail::ParallelFor3D< Device >::exec( startX, startY, startZ, endX, endY, endZ, launch_config, f, args... );
   }
};

}  // namespace Algorithms
}  // namespace TNL
