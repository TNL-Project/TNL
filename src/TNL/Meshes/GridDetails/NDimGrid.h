
#pragma once

#include <TNL/Logger.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <TNL/Meshes/GridDetails/Templates/Templates.h>

#include <type_traits>

namespace TNL {
namespace Meshes {

template <int Dimension, typename Real, typename Device, typename Index>
class NDimGrid {
  public:
   template <int ContainerDimension, typename ContainerValue, std::enable_if_t<(ContainerDimension > 0), bool> = true>
   using Container = TNL::Containers::StaticVector<ContainerDimension, ContainerValue>;

   using IndexType = Index;
   using DeviceType = Device;
   using RealType = Real;

   using Coordinate = Container<Dimension, Index>;
   using Point = Container<Dimension, Real>;

   ///////////////////////////////
   // Compatability with meshes
   using CoordinatesType = Container<Dimension, Index>;
   using PointType = Container<Dimension, Real>;
   using GlobalIndexType = Index;
   ///////////////////////////////

   using EntitiesCounts = Container<Dimension + 1, Index>;

   using OrientationBasesContainer = Container<1 << Dimension, Coordinate>;

   /**
    * @brief Returns the dimension of grid
    */
   static constexpr int getMeshDimension() { return Dimension; };
    /**
    * @brief Returns the coefficient powers size.
    */
   static constexpr int spaceStepsPowersSize = 5;

   using SpaceProductsContainer = Container<std::integral_constant<Index, Templates::pow(spaceStepsPowersSize, Dimension)>::value, Real>;

   NDimGrid() {
      Coordinate zero = 0;

      setDimensions(zero);

      fillBases();
   }

   /**
    * @brief Returns the number of orientations for entity dimension.
    *        For example in 2-D Grid the edge can be vertical or horizontal.
    */
   static constexpr Index getEntityOrientationsCount(const Index entityDimension);

   template <typename... Dimensions,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Dimensions>...>, bool> = true,
             std::enable_if_t<sizeof...(Dimensions) == Dimension, bool> = true>
   void setDimensions(Dimensions... dimensions);

   void setDimensions(const Container<Dimension, Index>& dimensions);

   /**
    * @brief - Returns dimensions as a count of edges along each axis
    * @param[in] index - Index of dimension
    */
   __cuda_callable__ inline Index getDimension(const Index index) const;
   /**
    * @brief - Returns dimensions as a count of edges along each axis
    * @param[in] indices - A list of indices
    */
   template <typename... DimensionIndex,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, DimensionIndex>...>, bool> = true,
             std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool> = true>
   __cuda_callable__ inline Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indices) const noexcept;
   /**
    * @brief - Returns dimensions as a count of edges along each axis
    */
   __cuda_callable__ inline const Coordinate& getDimensions() const noexcept;
   /**
    * @brief - Returns count of entities of specific dimension
    */
   __cuda_callable__ inline Index getEntitiesCount(const Index index) const;
   /**
    * @brief - Returns count of entities of specific dimension
    */
   template <int EntityDimension,
             std::enable_if_t<Templates::isInClosedInterval(0, EntityDimension, Dimension), bool> = true>
   __cuda_callable__ inline Index getEntitiesCount() const noexcept;
    /**
    * @brief - Returns count of entities of specific dimension
    */
   template <typename... DimensionIndex,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, DimensionIndex>...>, bool> = true,
             std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool> = true>
   __cuda_callable__ inline Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indices) const;
   /**
    * @brief - Returns count of entities of specific dimension
    */
   __cuda_callable__ inline const EntitiesCounts& getEntitiesCounts() const noexcept;
   /**
    * @brief - Returns count of entities of specific dimension and orientation
    *
    * @param[in] dimension
    * @param[in] orientation - orientation of the entity
    */
   __cuda_callable__ inline Index getOrientedEntitiesCount(const Index dimension, const Index orientation) const;
   /**
    * @brief - Returns count of entities of specific dimension and orientation
    *
    * @param[in] dimension
    * @param[in] orientation - orientation of the entity
    */
   template<int EntityDimension,
            int EntityOrientation,
            std::enable_if_t<Templates::isInClosedInterval(0, EntityDimension, Dimension), bool> = true,
            std::enable_if_t<Templates::isInClosedInterval(0, EntityOrientation, Dimension), bool> = true>
   __cuda_callable__ inline Index getOrientedEntitiesCount() const noexcept;
   /**
    * @brief - Returns basis of the entity with the specific orientation
    *
    * @param[in] dimension
    * @param[in] orientation - orientation of the entity
    */
   template<int EntityDimension>
   __cuda_callable__ inline Coordinate getBasis(Index orientation) const noexcept;
   /**
    * \brief Sets the origin and proportions of this grid.
    *
    * \param origin Point where this grid starts.
    * \param proportions Total length of this grid.
    */
   void setDomain(const Point& origin, const Point& proportions);
   /**
    * @brief Set the Origin of the grid
    */
   void setOrigin(const Point& origin) noexcept;
   /**
    * @brief Set the Origin of the grid
    */
   template <typename... Coordinates,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Coordinates>...>, bool> = true,
             std::enable_if_t<sizeof...(Coordinates) == Dimension, bool> = true>
   void setOrigin(Coordinates... coordinates) noexcept;
   /**
    * @brief - Returns the origin of the grid
    */
   __cuda_callable__ inline const Point& getOrigin() const noexcept;
   /**
    * @brief Set the Space Steps along each dimension of the grid
    */
   void setSpaceSteps(const Point& steps) noexcept;
   /**
    *  @brief - Specifies space steps of the grid
    */
   template <typename... Steps,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Steps>...>, bool> = true,
             std::enable_if_t<sizeof...(Steps) == Dimension, bool> = true>
   void setSpaceSteps(Steps... spaceSteps) noexcept;
   /**
    * @brief - Returns the space staps of the grid
    */
   __cuda_callable__ inline const Point& getSpaceSteps() const noexcept;
   /**
    * @brief Returns product of space steps
    */
   template <typename... Powers,
             std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Powers>...>, bool> = true,
             std::enable_if_t<sizeof...(Powers) == Dimension, bool> = true>
   __cuda_callable__ inline Real getSpaceStepsProducts(Powers... powers) const;
   /**
    * @brief Returns product of space steps
    */
   __cuda_callable__ inline Real getSpaceStepsProducts(const Coordinate& powers) const;
   /**
    * @brief Returns product of space step
    */
   template <Index... Powers,
             std::enable_if_t<sizeof...(Powers) == Dimension, bool> = true>
   __cuda_callable__ inline Real getSpaceStepsProducts() const noexcept;
   /**
    * @brief Get the Smalles Space Steps object
    */
   __cuda_callable__ inline Real getSmallestSpaceStep() const noexcept;
   /**
    * @brief Get the proportions of the Grid
    */
   __cuda_callable__ inline const Point& getProportions() const noexcept;
   /**
    * @brief Writes info about the grid
    */
   void writeProlog(TNL::Logger& logger) const noexcept;

  protected:
   Coordinate dimensions;
   /**
    * @brief - A list of elements count along specific directions.
    *          First, (n choose 0) elements will contain the count of 0
    *          dimension elements Second, (n choose 1) elements will contain the count
    *          of 1-dimension elements....
    *
    * For example, let's have a 3-d grid, then the map indexing will
    * be the next: 0 - 0 - count of vertices 1, 2, 3 - count of edges in x, y,
    * z plane 4, 5, 6 - count of faces in xy, yz, zy plane 7 - count of cells
    * in z y x plane
    *
    * @warning - The ordering of is lexigraphical.
    */
   Container<1 << Dimension, Index> entitiesCountAlongBases;
   /**
    * @brief - A cumulative map over dimensions.
    */
   Container<Dimension + 1, Index> cumulativeEntitiesCountAlongBases;

   Point origin, proportions, spaceSteps;

   OrientationBasesContainer bases;
   SpaceProductsContainer spaceStepsProducts;

   void fillEntitiesCount();
   void fillSpaceSteps();
   void fillSpaceStepsPowers();
   void fillProportions();
   void fillBases();

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseAll(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseAll(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseInterior(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseInterior(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseBoundary(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void traverseBoundary(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

   template <typename Func, typename... FuncArgs>
   void forEachPermutation(const Index k, const Index n, Func func, FuncArgs... args) const;
};
}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/NDimGrid.hpp>
