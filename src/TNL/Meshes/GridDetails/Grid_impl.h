
#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL
{
   namespace Meshes
   {

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Dimensions,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void Grid<Dimension, Real, Device, Index>::setDimensions(Dimensions... dimensions) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            TNL_ASSERT_GT(x, 0, "Dimension must be positive");
            this->dimensions[i] = x;
            i++;
         }

         fillEntitiesCount();
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::setDimensions(const Grid<Dimension, Real, Device, Index>::Container<Dimension, Index> &dimensions) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            TNL_ASSERT_GT(x, 0, "Dimension must be positive");
            this->dimensions[i] = x;
            i++;
         }

         fillEntitiesCount();
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          Index
          Grid<Dimension, Real, Device, Index>::getDimension(Index index) const noexcept
      {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LT(index, Dimension, "Index must be less than Dimension");

         return dimensions[index];
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... DimensionIndex,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      typename Grid<Dimension, Real, Device, Index>::Container<sizeof...(DimensionIndex), Index>
      Grid<Dimension, Real, Device, Index>::getDimensions(DimensionIndex... indices) const noexcept
      {
         Container<sizeof...(DimensionIndex), Index> result{indices...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this->getDimension(result[i]);

         return result;
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      Grid<Dimension, Real, Device, Index>::getDimensions() const noexcept
      {
         return this->dimensions;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          Index
          Grid<Dimension, Real, Device, Index>::getEntitiesCount(Index index) const noexcept
      {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LE(index, Dimension, "Index must be less than or equal to Dimension");

         return cumulativeDimensionMap(index);
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <int EntityDimension,
                typename = std::enable_if_t<(EntityDimension >= 0)>,
                typename = std::enable_if_t<(EntityDimension <= Dimension)>>
      __cuda_callable__
          Index
          getEntitiesCount() const noexcept
      {
         return this->cumulativeDimensionMap(EntityDimension);
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::setOrigin(const Container<Dimension, Index> &origin) noexcept
      {
         this->origin = origin;
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void Grid<Dimension, Real, Device, Index>::setOrigin(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->origin[i] = x;
            i++
         }
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          Index
          Grid<Dimension, Real, Device, Index>::getOrigin() const noexcept
      {
         return this->origin;
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillEntitiesCount() noexcept
      {
         std::array<bool, Dimension> combinationBuffer = {};
         std::size_t j = 0;

         for (std::size_t i = 0; i < Dimension + 1; i++)
            cumulativeEntitiesCountAlongBases[i] = 0;

         for (std::size_t i = 0; i <= Dimension; i++)
         {
            std::fill(combinationBuffer.begin(), combinationBuffer.end(), false);
            std::fill(combinationBuffer.end() - i, combinationBuffer.end(), true);

            do
            {
               int result = 1;

               for (std::size_t k = 0; k < combinationBuffer.size(); k++)
                  result *= combinationBuffer[k] ? dimensions[Dimension - k - 1] : dimensions[Dimension - k - 1] + 1;

               entitiesCountAlongBases[j] = result;
               cumulativeEntitiesCountAlongBases[i] += result;

               j++;
            } while (std::next_permutation(combinationBuffer.begin(), combinationBuffer.end()));
         }
      }

      //     // TODO: - Implement
      //     template <typename Dimension,
      //               typename Real,
      //               typename Device,
      //               typename Index>
      //     void Grid<Dimension, Real, Device, Index>::fillSpaceProducts() noexcept
      //     {
      //     }

      // TODO: - Implement
      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillProportions() noexcept
      {
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillSpaceSteps() noexcept
      {
      }
   }
}
