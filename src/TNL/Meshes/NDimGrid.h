
#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Logger.h>
#include <TNL/Algorithms/ParallelFor.h>

#include <type_traits>

namespace TNL {
namespace Meshes {

namespace Templates {
template <bool...>
struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true> >;

template <size_t Dimension, size_t Begin, size_t End, typename Func>
struct MetaFor {
  public:
   static constexpr void exec(Func &&function) {
      static_assert(Dimension > 0);

      for (size_t i = Begin; i != End; ++i) {
         auto bind_an_argument = [i, &function](auto... args) { function(i, args...); };

         MetaFor<Dimension - 1, Begin, End, Func>(std::forward(bind_an_argument));
      }
   }
};

template <size_t Begin, size_t End, typename Func>
struct MetaFor<1, Begin, End, Func> {
  public:
   static constexpr void exec(Func &&function) {
      for (size_t i = Begin; i != End; ++i) function(i);
   }
};

template <typename... Elements, typename ResultType, typename Func>
constexpr ResultType meta_reduce(ResultType &&initial, Func &&function, Elements... elements) {
   ResultType result = initial;

   for (const auto &element : {elements...}) result = function(result, element);

   return result;
}

template <size_t Value, size_t Power>
constexpr size_t pow() {
   size_t result = 1;

   for (size_t i = 0; i < Power; i++) {
      result *= Value;
   }

   return result;
}

}  // namespace Templates

// A base class for common methods for each grid.
template <int Dimension, typename Real = double, typename Device = Devices::Host, typename Index = int>
class NDimGrid {
  public:
   template <int ContainerDimension,
             typename ContainerValue,
             std::enable_if_t<(ContainerDimension > 0), bool> = true>
   using Container = TNL::Containers::StaticVector<ContainerDimension, ContainerValue>;

   using IndexType = Index;
   using DeviceType = Device;
   using RealType = Real;

   using Coordinate = Container<Dimension, Index>;
   using Point = Container<Dimension, Real>;

   /**
    * \brief Returns number of this mesh grid dimensions.
    */
   static constexpr int getMeshDimension() { return Dimension; };

   NDimGrid() {}
   // empty destructor is needed only to avoid crappy nvcc warnings
   virtual ~NDimGrid() {}
   /**
    *  @brief - Specifies dimensions of the grid as the number of edges at each dimenison
    */
   template <typename... Dimensions,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Index, Dimensions>::value...>::value, bool> = true,
             std::enable_if_t<sizeof...(Dimensions) == Dimension, bool> = true>
   void setDimensions(Dimensions... dimensions);
   /**
    *  @brief - Specifies dimensions of the grid
    */
   void setDimensions(const Container<Dimension, Index> &dimensions);
   /**
    * @param[in] index - Index of dimension
    */
   __cuda_callable__ inline
   Index getDimension(Index index) const;
   /**
    * @param[in] indices - A dimension indicies pack
    */
   template <typename... DimensionIndex,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Index, DimensionIndex>::value...>::value, bool> = true,
             std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool> = true >
   __cuda_callable__ inline
   Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indices) const noexcept;
   /**
    * @brief Get all dimensions of the objects
    */
   __cuda_callable__ inline
   const Coordinate& getDimensions() const noexcept;
   /**
    * @param[in] index - index of dimension
    */
   __cuda_callable__ inline
   Index getEntitiesCount(Index index) const;
   /**
    * @param[in] index - index of dimension
    */
   template <int EntityDimension,
             std::enable_if_t<(EntityDimension >= 0), bool> = true,
             std::enable_if_t<(EntityDimension <= Dimension), bool> = true>
   __cuda_callable__ inline
   Index getEntitiesCount() const noexcept;
   /**
    * @brief - Returns the number of entities of specific dimension
    */
   template <typename... DimensionIndex,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Index, DimensionIndex>::value...>::value, bool> = true,
             std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool> = true>
   __cuda_callable__ inline
   Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indices) const;
   /**
    * @brief - Returns entities counts along every dimension
    */
   __cuda_callable__ inline
   const Container<Dimension + 1, Index>& getEntitiesCounts() const noexcept;
   /**
    * \brief Sets the origin and proportions of this grid.
    * \param origin Point where this grid starts.
    * \param proportions Total length of this grid.
    */
   void setDomain(const Point &origin, const Point &proportions);
   /**
    * @brief Set the Origin of the grid
    */
   void setOrigin(const Point &origin) noexcept;
   /**
    *  @brief - Specifies dimensions of the grid
    *  @param[in] coordinates - A parameter pack, which specifies points count
    * in the specific coordinates. Most significant dimension is in the
    * beginning of the list. Least significant dimension is in the end of the
    * list
    */
   template <typename... Coordinates,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Real, Coordinates>::value...>::value, bool> = true,
             std::enable_if_t<sizeof...(Coordinates) == Dimension, bool> = true>
   void setOrigin(Coordinates... coordinates) noexcept;
   /**
    * @brief - Returns the origin of the grid
    */
   __cuda_callable__ inline const Point& getOrigin() const noexcept;
   /**
    * @brief Set the Space Steps along each dimension of the grid
    */
   void setSpaceSteps(const Point &steps) noexcept;
   /**
    *  @brief - Specifies space steps of the grid
    *  @param[in] coordinates - A parameter pack, which specifies space steps
    * in the specific coordinates. Most significant dimension is in the
    * beginning of the list. Least significant dimension is in the end of the
    * list
    */
   template <typename... Coordinates,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Real, Coordinates>::value...>::value, bool> = true,
             std::enable_if_t<sizeof...(Coordinates) == Dimension, bool> = true>
   void setSpaceSteps(Coordinates... coordinates) noexcept;
   /**
    * @brief - Returns the origin of the grid
    */
   __cuda_callable__ inline
   const Point& getSpaceSteps() const noexcept;
   /**
    * @brief Returns product of space steps to the xPow.
    */
   template <typename... Powers,
             std::enable_if_t<Templates::conjunction<std::is_convertible<Real, Powers>::value...>::value, bool> = true,
             std::enable_if_t<sizeof...(Powers) == Dimension, bool> = true>
   __cuda_callable__ inline
   Real getSpaceStepsProducts(Powers... powers) const noexcept;
   /**
    * @brief Get the Smalles Space Steps object
    */
   __cuda_callable__ inline
   Real getSmallestSpaceSteps() const noexcept;
   /**
    * @brief Get the proportions of the Grid
    */
   __cuda_callable__ inline
    const Point& getProportions() const noexcept;
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
   void writeProlog(Logger &&logger) const noexcept;
  protected:
   static constexpr int spaceStepsPowersSize = 5;

   Coordinate dimensions;
   /**
    * @brief - A list of elements count along specific directions.
    *          First, (n choose 0) elements will contain the count of 0
    * dimension elements Second, (n choose 1) elements will contain the count
    * of 1-dimension elements
    *          ....
    *
    *          For example, let's have a 3-d grid, then the map indexing will
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
   /**
    * @brief - Origin and proportions of the grid domain
    */
   Point origin, proportions, spaceSteps;

   Container<std::integral_constant<Index, Templates::pow<spaceStepsPowersSize, Dimension>()>::value, Real> spaceProducts;

   void fillEntitiesCount();
   void fillSpaceSteps();
   void fillSpaceStepsPowers();
   void fillProportions();

   template<typename Func, typename... FuncArgs>
   void forEach(const Container<Dimension, Index> from, const Container<Dimension, Index> to, Func func, FuncArgs... args);
};
}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/NDimGrid_impl.h>
