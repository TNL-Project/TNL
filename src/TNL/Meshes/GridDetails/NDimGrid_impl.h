
#pragma once

#include <TNL/Meshes/NDimGrid.h>

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
      void Grid<Dimension, Real, Device, Index>::setDimensions(Dimensions... dimensions)
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            TNL_ASSERT_GT(x, 0, "Dimension must be positive");
            this->dimensions[i] = x;
            i++;
         }

         fillEntitiesCount();
         fillSpaceSteps();
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
         fillSpaceSteps();
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
      Grid<Dimension, Real, Device, Index>::Container<sizeof...(DimensionIndex), Index>
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

      template <int Dimension,
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

      template <typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::setDomain(const Container<Dimension, Index> &origin,
                                                           const Container<Dimension, Index> &proportions)
      {
         this->origin = origin;
         this->proportions = proportions;

         this -> fillSpaceSteps();
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
            i++;
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
      void Grid<Dimension, Real, Device, Index>::setSpaceSteps(const Container<Dimension, Real> &spaceSteps) noexcept
      {
         this->spaceSteps = spaceSteps;

         fillSpaceStepsPowers();
         fillProportions();
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Real, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void Grid<Dimension, Real, Device, Index>::setSpaceSteps(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->spaceSteps[i] = x;
            i++;
         }

         fillSpaceStepsPowers();
         fillProportions();
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          Index
          Grid<Dimension, Real, Device, Index>::getSpaceSteps() const noexcept
      {
         return this->spaceSteps;
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
      Grid<Dimension, Real, Device, Index>::Container<Dimension, Real>
      Grid<Dimension, Real, Device, Index>::getProportions() const noexcept {
         return this -> proportions;
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Powers,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Powers>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Powers) == Dimension>>
      __cuda_callable__
      Real Grid<Dimension, Real, Device, Index>::getSpaceStepsProducts(Powers... powers) const noexcept {
         constexpr halfSize = this -> spaceStepsPowersSize >> 1;

         Index i = 0;
         Index base = 1;

         for (auto x: {powers...}) {
            static_assert(x >= -halfSize && x <= halfSize; "Unsupported size of the powers");

            i += x * base;

            base *= this -> spaceStepsPowersSize;
         }

         return this -> spaceStepsProducts(i);
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__ inline Real Grid<Dimension, Real, Device, Index>::getSmallestSpaceSteps() const noexcept
      {
         Real minStep = this->spaceSteps[0];
         Index i = 1;

         while (i != Dimension)
            minStep = min(minStep, this->spaceSteps[i++]);

         return minStep;
      }

      void Grid<Dimension, Real, Device, Index>::writeProlog(Logger && logger) const noexcept {
         logger.writeParameter("Dimensions:", this -> dimensions);

         logger.writeParameter("Origin:", this -> origin);
         logger.writeParameter("Proportions:", this -> proportions);
         logger.writeParameter("Space steps:", this -> spaceSteps);

         for (Index i = 0; i <= Dimension; i++) {
            String tmp = String("Entities count along dimension ") + String(i) + ":";

            logger.writeParameter(tmp, this -> cumulativeEntitiesCountAlongBases[i];)
         }
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillEntitiesCount()
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

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillProportions()
      {
         Index i = 0;

         while (i != Dimension) {
            this->proportions[i] = this->spaceSteps[i] * this->dimensions[i];
            i++;
         }
      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillSpaceSteps()
      {
         bool hasAnyInvalidDimension = false;

         for (Index i = 0; i < Dimension; i++) {
            if (this -> dimensions[i] <= 0) {
               hasAnyInvalidDimension = true;
               break;
            }
         }

         if (!hasAnyInvalidDimension) {
            for (Index i = 0; i < Dimension; i++)
               this -> spaceSteps[i] = this -> proportions[i] / this -> dimensions[i];

            fillSpaceStepsPowers();
         }

      }

      template <int Dimension,
                typename Real,
                typename Device,
                typename Index>
      void Grid<Dimension, Real, Device, Index>::fillSpaceStepsPowers()
      {
         Container<spaceStepsPowersSize * EntityDimension, Real> powers;

         for (Index i = 0; i < EntityDimension; i++)
            for (Index j = 0, int power = -2; j < spaceStepsPowersSize; j++, power++)
               powers[i * 5 + j] = pow(this->spaceSteps[i], power);

         for (Index i = 0; i < this -> spaceStepsPowers.getSize(); i++) {
            Real product = 1;
            Index index = i;

            for (Index j = 0; j < Dimension; j++) {
               Index residual = index % divider;

               index /= this -> spaceStepsPowersSize;

               product *= powers[residual];
            }

            powers[i] = product;
         }
      }
   }
}
