// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Containers/StaticArray.h>

#include <type_traits>

namespace TNL
{
   namespace Meshes
   {

      namespace Templates
      {
         template <bool...>
         struct bool_pack
         {
         };

         template <bool... Bs>
         using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

         template <size_t Dimension, size_t Begin, size_t End, typename Func>
         struct MetaFor
         {
            public:
               static constexpr void exec(Func && function)
               {
                  static_assert(Dimension > 0);

                  for (size_t i = Begin; i != End; ++i)
                  {
                     auto bind_an_argument = [i, &function](auto... args)
                     {
                        function(i, args...);
                     };

                     MetaFor<Dimension - 1, Begin, End, Func>(std::forward(bind_an_argument));
                  }
               }
         };

         template <size_t Begin, size_t End, typename Func>
         struct MetaFor<1, Begin, End, Func>
         {
            public:
               static constexpr void exec(Func && function)
               {
                  for (size_t i = Begin; i != End; ++i)
                     function(i);
               }
         };

         template <typename... Elements, typename ResultType, typename Func>
         constexpr ResultType meta_reduce(ResultType &&initial, Func &&function, Elements... elements)
         {
            ResultType result = initial;

            for (const auto &element : {elements...})
               result = function(result, element);

            return result;
         }
      }

      template <int Dimension,
                typename Real = double,
                typename Device = Devices::Host,
                typename Index = int>
      class Grid
      {
      public:
         template <int ContainerDimension,
                   typename ContainerValue,
                   typename = std::enable_if_t<(Dimension > 0)>,
                   typename = std::enable_if_t<std::is_integral<Index>::value>>
         using Container = TNL::Containers::StaticArray<ContainerDimension, ContainerValue>;

         /**
          * \brief Returns number of this mesh grid dimensions.
          */
         static constexpr int getMeshDimension() { return Dimension; };

         // empty destructor is needed only to avoid crappy nvcc warnings
         ~Grid() {}
         /**
          *  @brief - Specifies dimensions of the grid
          *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
          *                          Most significant dimension is in the beginning of the list.
          *                          Least significant dimension is in the end of the list
          */
         template <typename... Dimensions,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                   typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
         void setDimensions(Dimensions... dimensions);
         /**
          *  @brief - Specifies dimensions of the grid
          *  @param[in] dimensions - A container with the dimension items
          */
         void setDimensions(const Container<Dimension, Index> &dimensions) noexcept;
         /**
          * @param[in] index - index of dimension
          */
         __cuda_callable__
             Index
             getDimension(Index index) const noexcept;
         /**
          * @param[in] indices - A dimension index pack
          */
         template <typename... DimensionIndex,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                   typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
         Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indices) const noexcept;
         /**
          * @brief Get all dimensions of the objects
          *
          * @return Container<Dimension, Index>
          */
         Container<Dimension, Index> getDimensions() const noexcept;
         /**
          * @param[in] index - index of dimension
          */
         __cuda_callable__
             Index
             getEntitiesCount(Index index) const noexcept;
         /**
          * @param[in] index - index of dimension
          */
         template <int EntityDimension,
                   typename = std::enable_if_t<(EntityDimension >= 0)>,
                   typename = std::enable_if_t<(EntityDimension <= Dimension)>>
         __cuda_callable__
             Index
             getEntitiesCount() const noexcept;
         /**
          * @brief - Returns the number of entities of specific dimension
          */
         template <typename... DimensionIndex,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                   typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
         Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indices) const noexcept;
         /**
          * \brief Sets the origin and proportions of this grid.
          * \param origin Point where this grid starts.
          * \param proportions Total length of this grid.
          */
         void setDomain(const Container<Dimension, Index> &origin,
                        const Container<Dimension, Index> &proportions);
         /**
          * @brief Set the Origin of the grid
          */
         void setOrigin(const Container<Dimension, Index> &origin) noexcept;
         /**
          *  @brief - Specifies dimensions of the grid
          *  @param[in] coordinates - A parameter pack, which specifies points count in the specific coordinates.
          *                           Most significant dimension is in the beginning of the list.
          *                           Least significant dimension is in the end of the list
          */
         template <typename... Coordinates,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                   typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
         void setOrigin(Coordinates... coordinates) noexcept;
         /**
          * @brief - Returns the origin of the grid
          */
         Container<Dimension, Index> getOrigin() const noexcept;
         /**
          * @brief Set the Space Steps along each dimension of the grid
          */
         void setSpaceSteps(const Container<Dimension, Real> &steps) noexcept;
         /**
          *  @brief - Specifies space steps of the grid
          *  @param[in] coordinates - A parameter pack, which specifies space steps in the specific coordinates.
          *                           Most significant dimension is in the beginning of the list.
          *                           Least significant dimension is in the end of the list
          */
         template <typename... Coordinates,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Real, Coordinates>::value...>::value>,
                   typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
         void setSpaceSteps(Coordinates... coordinates) noexcept;
         /**
          * @brief - Returns the origin of the grid
          */
         Container<Dimension, Real> getSpaceSteps() const noexcept;
         /**
          * @brief Returns product of space steps to the xPow.
          */
         template <typename... Powers,
                   typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Powers>::value...>::value>,
                   typename = std::enable_if_t<sizeof...(Powers) == Dimension>>
         __cuda_callable__
             Real
             getSpaceStepsProducts(Powers... powers) const noexcept;
         /**
          * @brief Get the Smalles Space Steps object
          */
         __cuda_callable__ inline Real getSmallesSpaceSteps() const noexcept;
         /**
          * @brief Get the proportions of the Grid
          */
         __cuda_callable__ Container<Dimension, Real> getProportions() const noexcept;
         /*
          * @brief Traverses all elements
          */
         template <int EntityDimension, typename Func, typename... FuncArgs>
         void forAll(Func func, FuncArgs... args) const;
         /*
          * @brief Traverses interior elements
          */
         template <int EntityDimension, typename Func, typename... FuncArgs>
         void forInterior(Func func, FuncArgs... args) const;
         /*
          * @brief Traverses boundary elements
          */
         template <int EntityDimension, typename Func, typename... FuncArgs>
         void forBoundary(Func func, FuncArgs... args) const;
         /**
          * @brief Writes info about the grid
          */
         void writeProlog();
      protected:
         static constexpr int spaceStepsPowersSize = 5;
         /**
          * @brief - Dimensions of the grid in the amount of edges for each axia.
          */
         Container<Dimension, Index> dimensions;
         /**
          * @brief - A list of elements count along specific directions.
          *          First, (n choose 0) elements will contain the count of 0 dimension elements
          *          Second, (n choose 1) elements will contain the count of 1-dimension elements
          *          ....
          *
          *          For example, let's have a 3-d grid, then the map indexing will be the next:
          *            0 - 0 - count of vertices
          *            1, 2, 3 - count of edges in x, y, z plane
          *            4, 5, 6 - count of faces in xy, yz, zy plane
          *            7 - count of cells in z y x plane
          *
          * @warning - The ordering of is lexigraphical.
          */
         Container<1 << Dimension, Index> entitiesCountAlongBases;
         /**
          * @brief - A cumulative map over dimensions.
          */
         Container<Dimension + 1, Index> cumulativeEntitiesCountAlongBases;
         /**
          * @brief - Origin and proportions of the grid domain
          */
         Container<Dimension, Index> origin, proportions;
         /**
          * @brief - Space steps along dimensions
          */
         Container<Dimension, Real> spaceSteps;

         Container<std::integral_constant<Index, pow(spaceStepsPowersSize, Dimension)>::value, Real> spaceProducts;

         void fillEntitiesCount();
         void fillSpaceSteps();
         void fillSpaceStepsPowers();
         void fillProportions();
      };

      // template< int Dimension, typename Real, typename Device, typename Index >
      // bool operator==( const Grid< Dimension, Real, Device, Index >& lhs,
      //                  const Grid< Dimension, Real, Device, Index >& rhs )
      // {
      //    return lhs.getDimensions() == rhs.getDimensions()
      //        && lhs.getOrigin() == rhs.getOrigin()
      //        && lhs.getProportions() == rhs.getProportions();
      // }

      // template< int Dimension, typename Real, typename Device, typename Index >
      // bool operator!=( const Grid< Dimension, Real, Device, Index >& lhs,
      //                  const Grid< Dimension, Real, Device, Index >& rhs )
      // {
      //    return ! (lhs == rhs);
      // }

      // template< int Dimension, typename Real, typename Device, typename Index >
      // std::ostream& operator<<( std::ostream& str, const Grid< Dimension, Real, Device, Index >& grid )
      // {
      //    str << "Grid dimensions:    " << grid.getDimensions()  << std::endl;
      //    str << "     origin:        " << grid.getOrigin()      << std::endl;
      //    str << "     proportions:   " << grid.getProportions() << std::endl;
      //    str << "     localBegin:    " << grid.getLocalBegin() << std::endl;
      //    str << "     localEnd:      " << grid.getLocalEnd() << std::endl;
      //    str << "     interiorBegin: " << grid.getInteriorBegin() << std::endl;
      //    str << "     interiorEnd:   " << grid.getInteriorEnd() << std::endl;
      //    return str;
      // }

   } // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
