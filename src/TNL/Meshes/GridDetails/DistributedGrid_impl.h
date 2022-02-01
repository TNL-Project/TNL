
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
      template <typename... Dimensions,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalBegin(Dimensions... point) noexcept
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
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalBegin(const Container<Dimension, Index> &point) noexcept
      {
         this->localBegin = point;
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
      template <typename... Dimensions,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalEnd(Dimensions... point) noexcept
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
      void DistributedGrid<Dimension, Real, Device, Index>::setLocalEnd(const Container<Dimension, Index> &point) noexcept
      {
         this->localEnd = point;
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
      template <typename... Dimensions,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorBegin(Dimensions... point) noexcept
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
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorBegin(const Container<Dimension, Index> &point) noexcept
      {
         this->interiorBegin = point;
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
      template <typename... Dimensions,
                typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorEnd(Dimensions... point) noexcept
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
      void DistributedGrid<Dimension, Real, Device, Index>::setInteriorEnd(const Container<Dimension, Index> &point) noexcept
      {
         this->interiorEnd = point;
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
