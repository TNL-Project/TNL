
#include <TNL/Meshes/DistributedGrid.h>

namespace TNL
{
   namespace Meshes
   {
      // TODO: Add checks

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalBegin(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->localBegin[i] = x;
            i++
         }
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalBegin(const Container<Dimension, Index> &coordinates) noexcept
      {
         this->localBegin = coordinates;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          DistributedGrid<Dimension, Real, Device, Index>::Container<Dimension, Index>
          DistributedGrid<Dimension, Real, Device, Index>::getLocalBegin() const
      {
         return this->localBegin;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalEnd(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->localEnd[i] = x;
            i++
         }
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalEnd(const Container<Dimension, Index> &coordinates) noexcept
      {
         this->localEnd = coordinates;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          DistributedGrid<Dimension, Real, Device, Index>::Container<Dimension, Index>
          DistributedGrid<Dimension, Real, Device, Index>::getLocalEnd() const
      {
         return this->localEnd;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorBegin(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->interiorBegin[i] = x;
            i++
         }
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorBegin(const Container<Dimension, Index> &coordinates) noexcept
      {
         this->interiorBegin = coordinates;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          DistributedGrid<Dimension, Real, Device, Index>::Container<Dimension, Index>
          DistributedGrid<Dimension, Real, Device, Index>::getInteriorBegin() const
      {
         return this->interiorBegin;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      template <typename... Coordinates,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorEnd(Coordinates... coordinates) noexcept
      {
         Index i = 0;

         for (auto x : {dimensions...})
         {
            this->interiorEnd[i] = x;
            i++
         }
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorEnd(const Container<Dimension, Index> &coordinates) noexcept
      {
         this->interiorEnd = coordinates;
      }

      template <typename Dimension,
                typename Real,
                typename Device,
                typename Index>
      __cuda_callable__
          DistributedGrid<Dimension, Real, Device, Index>::Container<Dimension, Index>
          DistributedGrid<Dimension, Real, Device, Index>::getInteriorEnd() const
      {
         return this->interiorEnd;
      }
   }
}
