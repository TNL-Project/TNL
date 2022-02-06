
#include <TNL/Devices/Host.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Meshes/Grid.h>

namespace TNL
{
   namespace Meshes
   {
      template <int Dimension,
                typename Real = double,
                typename Device = Devices::Host,
                typename Index = int>
      class DistributedGrid: public NDimGrid<Dimension, Real, Device, Index> {
         public:
            template <int ContainerDimension,
                      typename ContainerIndex>
            using Container = typename Grid<Dimension, Real, Device, Index>::Container<ContainerDimension, ContainerIndex>;
            /**
             *  @brief - Specifies coordinates of the local grid
             *  @param[in] dimensions - A parameter pack, which specifies coordinates count in the specific dimension.
             *                          Most significant dimension is in the beginning of the list.
             *                          Least significant dimension is in the end of the list
             */
            template <typename... Coordinates,
                      typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                      typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
            void setLocalBegin(Coordinates... coordinates) noexcept;
            /**
             *  @brief - Specifies coordinates of the local grid
             *  @param[in] dimensions - A container with the point coordinates
             */
            void setLocalBegin(const Container<Dimension, Index> &coordinates) noexcept;
            /**
             * @brief Get the Local Begin object
             */
            __cuda_callable__
            Container<Dimension, Index> getLocalBegin() const;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
             *                          Most significant dimension is in the beginning of the list.
             *                          Least significant dimension is in the end of the list
             */
            template <typename... Coordinates,
                      typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                      typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
            void setLocalEnd(Coordinates... coordinates) noexcept;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A container with the dimension items
             */
            void setLocalEnd(const Container<Dimension, Index> &coordinates) noexcept;
            /**
             * @brief Get the Local end object
             */
            __cuda_callable__
            Container<Dimension, Index> getLocalEnd() const;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
             *                          Most significant dimension is in the beginning of the list.
             *                          Least significant dimension is in the end of the list
             */
            template <typename... Coordinates,
                      typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                      typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
            void setInteriorBegin(Coordinates... point) noexcept;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A container with the dimension items
             */
            void setInteriorBegin(const Container<Dimension, Index> &point) noexcept;
            /**
             * @brief Get the Interior Begin object
             */
            __cuda_callable__
            Container<Dimension, Index> getInteriorBegin() const;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
             *                          Most significant dimension is in the beginning of the list.
             *                          Least significant dimension is in the end of the list
             */
            template <typename... Coordinates,
                      typename = std::enable_if_t<Templates::conjunction<std::is_same<Index, Coordinates>::value...>::value>,
                      typename = std::enable_if_t<sizeof...(Coordinates) == Dimension>>
            void setInteriorEnd(Coordinates... coordinates) noexcept;
            /**
             *  @brief - Specifies dimensions of the grid
             *  @param[in] dimensions - A container with the dimension items
             */
            void setInteriorEnd(const Container<Dimension, Index> &coordinates) noexcept;
            /**
             * @brief Get the Interior End object
             */
            __cuda_callable__
            Container<Dimension, Index> getInteriorEnd() const;
         protected:
            Container<Dimension, Index> localBegin, localEnd, interiorBegin, interiorEnd;
      };
   }
}
