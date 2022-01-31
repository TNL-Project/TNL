
#pragma once

#include <array>
#include <type_traits>

namespace TNL
{
  namespace Meshes
  {

    namespace Utils {
      template<typename Container, int Size> struct NDimensionalContainer;

      // TODO: - Template defined container
      // TODO: - Template defined for loop
      // TODO: - Template defined index
    }

    template <typename Dimension,
              typename Real,
              typename Device,
              typename Index>
    class Grid
    {
    public:
      template <int Dimension,
                typename Index,
                typename = std::enable_if_t<(Dimension > 0)>,
                typename = std::enable_if_t<std::is_integral<Index>::value>>
      using Container = TNL::Containers::StaticArray<Dimension, Index>;

      virtual ~Grid(){};
    protected:
      /**
       *  @brief - Specifies dimensions of the grid
       *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
       *                          Most significant dimension is in the beginning of the list.
       *                          Least significant dimension is in the end of the list
       */
      template <typename... Dimensions,
                typename = std::enable_if_t<conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void setDimensions(Dimensions... dimensions) noexcept;
      /**
       * @param[in] index - index of dimension
       */
      __cuda_callable__
      Index getDimension(Index index) const noexcept;
      /**
       * @param[in] indices - A dimension index pack
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indices) const noexcept;
      /**
       * @param[in] index - index of dimension
       */
      __cuda_callable__
      Index getEntitiesCount(Index index) const noexcept;
      /**
       * @brief - Returns the number of entities of specific dimension
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indices) const noexcept;
    private:
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

      // TODO, Think how to generate containers using meta programming
      Container<5, Container<5, Container<5, Real>>> spaceProducts;

      /**
       * @brief A proportion is the product of the dimension size and the space product.
       */
      Container<Dimension, Real> proportions;

      void fillEntitiesCount() noexcept;
      // TODO: Think how to generate for loops based on the dimension.
      void fillSpaceProducts() noexcept;
      void fillProportions() noexcept;
    };

  }
}