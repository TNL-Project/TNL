// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>

#include <TNL/Containers/NDArrayView.h>

namespace TNL::Containers {

/**
 * \defgroup ndarray  N-dimensional arrays
 *
 * This group includes several classes for the representation of N-dimensional
 * arrays and some helper classes. See the \ref ug_NDArrays "Users' Guide" for
 * showing how these data structures can be used.
 *
 * @{
 */

/**
 * \brief Base storage class for \ref NDArray and \ref StaticNDArray.
 *
 * \tparam Type of the underlying one-dimensional array for storing the elements.
 * \tparam Type of the N-dimensional indexer, \ref NDArrayIndexer.
 * \tparam Type of the \ref TNL::Devices "device" used for running operations on the array.
 *
 * Note that the class inherits from the `Indexer`, i.e. \ref NDArrayIndexer.
 *
 * See also the \ref ug_NDArrays "Users' Guide".
 */
template< typename Array, typename Indexer, typename Permutation, typename Device = typename Array::DeviceType >
class NDArrayStorage : public Indexer
{
public:
   //! \brief Type of the values stored in the array.
   using ValueType = typename Array::ValueType;

   //! \brief Type of the \ref TNL::Devices "device" used for running operations on the array.
   using DeviceType = Device;

   //! \brief Type of indices used for addressing the array elements.
   using IndexType = typename Indexer::IndexType;

   //! \brief Type of the underlying object which represents the sizes of the N-dimensional array.
   using SizesHolderType = typename Indexer::SizesHolderType;

   //! \brief Type of the underlying object which represents the strides of the N-dimensional array.
   using StridesHolderType = typename Indexer::StridesHolderType;

   //! \brief Permutation that determines the internal memory layout of the N-dimensional array.
   using PermutationType = Permutation;

   //! \brief Sequence of integers representing the overlaps in each dimension
   //! of a distributed N-dimensional array.
   using OverlapsType = typename Indexer::OverlapsType;

   //! \brief Type of the N-dimensional indexer, \ref NDArrayIndexer.
   using IndexerType = Indexer;

   //! Compatible \ref NDArrayView type.
   using ViewType = NDArrayView< ValueType, DeviceType, IndexerType, PermutationType >;

   //! Compatible constant \ref NDArrayView type.
   using ConstViewType = NDArrayView< std::add_const_t< ValueType >, DeviceType, IndexerType, PermutationType >;

   //! \brief View type of the underlying one-dimensional array storing the elements.
   using StorageArrayViewType = typename ViewType::StorageArrayViewType;

   //! \brief Constant view type of the underlying one-dimensional array storing the elements.
   using ConstStorageArrayViewType = typename ViewType::ConstStorageArrayViewType;

   //! \brief Constructs an empty storage with zero size.
   NDArrayStorage() = default;

   //! \brief Copy constructor (makes a deep copy).
   explicit NDArrayStorage( const NDArrayStorage& ) = default;

   //! \brief Move constructor for initialization from \e rvalues.
   NDArrayStorage( NDArrayStorage&& ) noexcept = default;

   //! \brief Copy-assignment operator for deep-copying data from another array.
   //! Mismatched sizes cause reallocations.
   NDArrayStorage&
   operator=( const NDArrayStorage& other ) = default;

   //! \brief Move-assignment operator for acquiring data from \e rvalues.
   NDArrayStorage&
   operator=( NDArrayStorage&& ) noexcept( false ) = default;

   //! \brief Templated copy-assignment operator for deep-copying data from another array.
   template< typename OtherArray >
   NDArrayStorage&
   operator=( const OtherArray& other )
   {
      static_assert( std::is_same_v< PermutationType, typename OtherArray::PermutationType >,
                     "Arrays must have the same permutation of indices." );
      // update sizes and strides
      detail::SetSizesCopyHelper< SizesHolderType, typename OtherArray::SizesHolderType >::copy( getSizes(), other.getSizes() );
      detail::SetSizesCopyHelper< StridesHolderType, typename OtherArray::StridesHolderType >::copy( getStrides(),
                                                                                                     other.getStrides() );
      // (re)allocate storage if necessary
      array.setSize( getStorageSize() );
      // copy data
      getView() = other.getConstView();
      return *this;
   }

   //! \brief Compares the array with another N-dimensional array.
   [[nodiscard]] bool
   operator==( const NDArrayStorage& other ) const
   {
      // TODO: contiguity check
      return getSizes() == other.getSizes() && getStrides() == other.getStrides() && array == other.array;
   }

   //! \brief Compares the array with another N-dimensional array.
   [[nodiscard]] bool
   operator!=( const NDArrayStorage& other ) const
   {
      // TODO: contiguity check
      return getSizes() != other.getSizes() || getStrides() != other.getStrides() || array != other.array;
   }

   /**
    * \brief Resets the array to the empty state.
    *
    * The current data will be deallocated, thus all pointers and views to
    * the array elements will become invalid.
    */
   void
   reset()
   {
      getSizes() = SizesHolderType{};
      detail::compute_dynamic_strides< PermutationType >( getStrides(), getSizes(), getOverlaps() );
      TNL_ASSERT_EQ( getStorageSize(), 0, "Failed to reset the sizes." );
      array.reset();
   }

   /**
    * \brief Returns a raw pointer to the data.
    *
    * This method can be called from device kernels.
    */
   [[nodiscard]] __cuda_callable__
   ValueType*
   getData()
   {
      return array.getData();
   }

   /**
    * \brief Returns a \e const-qualified raw pointer to the data.
    *
    * This method can be called from device kernels.
    */
   [[nodiscard]] __cuda_callable__
   std::add_const_t< ValueType >*
   getData() const
   {
      return array.getData();
   }

   // methods from the base class
   using IndexerType::getDimension;
   using IndexerType::getOverlap;
   using IndexerType::getOverlaps;
   using IndexerType::getSize;
   using IndexerType::getSizes;
   using IndexerType::getStorageIndex;
   using IndexerType::getStorageSize;
   using IndexerType::getStride;
   using IndexerType::getStrides;
   using IndexerType::isContiguousBlock;

   //! Returns a const-qualified reference to the underlying indexer.
   [[nodiscard]] __cuda_callable__
   const IndexerType&
   getIndexer() const
   {
      return *this;
   }

   //! \brief Returns a modifiable view of the array.
   [[nodiscard]] __cuda_callable__
   ViewType
   getView()
   {
      return ViewType( array.getData(), getIndexer() );
   }

   //! \brief Returns a non-modifiable view of the array.
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const
   {
      return ConstViewType( array.getData(), getIndexer() );
   }

   //! \brief Returns a modifiable view of the underlying storage array.
   [[nodiscard]] StorageArrayViewType
   getStorageArrayView()
   {
      return getView().getStorageArrayView();
   }

   //! \brief Returns a non-modifiable view of the underlying storage array.
   [[nodiscard]] ConstStorageArrayViewType
   getConstStorageArrayView() const
   {
      return getConstView().getConstStorageArrayView();
   }

   /**
    * \brief Returns a modifiable view of a subarray.
    *
    * \tparam Dimensions Sequence of integers representing the dimensions of the
    *                    array which selects the subset of dimensions to appear
    *                    in the subarray.
    * \param indices Indices of the _origin_ of the subarray in the whole array.
    *                The number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns \ref NDArrayView instantiated for \ref ValueType, \ref DeviceType
    *          and an \ref NDArrayIndexer matching the specified dimensions and
    *          subarray sizes.
    */
   template< std::size_t... Dimensions, typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   auto
   getSubarrayView( IndexTypes&&... indices )
   {
      return getView().template getSubarrayView< Dimensions... >( std::forward< IndexTypes >( indices )... );
   }

   /**
    * \brief A "safe" accessor for array elements.
    *
    * It can be called only from the host code and it will do a slow copy from
    * the device.
    */
   template< typename... IndexTypes >
   [[nodiscard]] ValueType
   getElement( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      const IndexType storageIndex = getStorageIndex( std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( storageIndex, array.getSize(), "storage index out of bounds" );
      return array.getElement( storageIndex );
   }

   /**
    * \brief Accesses an element of the array.
    *
    * \param indices Indices of the element in the N-dimensional array. The
    *                number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns Reference to the array element.
    */
   template< typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      const IndexType storageIndex = getStorageIndex( std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( storageIndex, array.getSize(), "storage index out of bounds" );
      return array[ storageIndex ];
   }

   /**
    * \brief Accesses an element of the array.
    *
    * \param indices Indices of the element in the N-dimensional array. The
    *                number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns Constant reference to the array element.
    */
   template< typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      const IndexType storageIndex = getStorageIndex( std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( storageIndex, array.getSize(), "storage index out of bounds" );
      return array[ storageIndex ];
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Index of the element in the one-dimensional array.
    * \returns Reference to the array element.
    */
   [[nodiscard]] __cuda_callable__
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInBounds( getSizes(), getOverlaps(), std::forward< IndexType >( index ) );
      return array[ index ];
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Index of the element in the one-dimensional array.
    * \returns Constant reference to the array element.
    */
   [[nodiscard]] __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInBounds( getSizes(), getOverlaps(), std::forward< IndexType >( index ) );
      return array[ index ];
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the array.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements. For example,
    * given a 3D array `a` with `int` as \ref IndexType, the function can be
    * defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forAll( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forAll( Func f,
           const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      dispatch( Begins{}, getSizes(), launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all internal elements of the array.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements. For example,
    * given a 3D array `a` with `int` as \ref IndexType, the function can be
    * defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forAll( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forInterior(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using Ends = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      using NoOverlapsType = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      detail::SetSizesSubtractHelper< 1, Ends, SizesHolderType, NoOverlapsType >::subtract(
         ends, getSizes(), NoOverlapsType{} );
      dispatch( Begins{}, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array inside the N-dimensional interval `[begins, ends)`.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements.
    */
   template< typename Device2 = DeviceType, typename Begins, typename Ends, typename Func >
   void
   forInterior(
      const Begins& begins,
      const Ends& ends,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "begins <= sizes", "ends <= sizes"
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all boundary elements of the array.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements. For example,
    * given a 3D array `a` with `int` as \ref IndexType, the function can be
    * defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forAll( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forBoundary(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      using SkipBegins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using SkipEnds = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      using NoOverlapsType = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      detail::SetSizesSubtractHelper< 1, SkipEnds, SizesHolderType, NoOverlapsType >::subtract(
         skipEnds, getSizes(), NoOverlapsType{} );

      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, SkipBegins{}, skipEnds, getSizes(), launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array outside the N-dimensional interval `[skipBegins, skipEnds)`.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements.
    */
   template< typename Device2 = DeviceType, typename SkipBegins, typename SkipEnds, typename Func >
   void
   forBoundary(
      const SkipBegins& skipBegins,
      const SkipEnds& skipEnds,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "skipBegins <= sizes", "skipEnds <= sizes"
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, skipBegins, skipEnds, getSizes(), launch_configuration, f );
   }

   // TODO: rename to setSizes and make sure that overloading with the following method works
   //! \brief Sets sizes of the array using an instance of \ref SizesHolder.
   void
   setSize( const SizesHolderType& sizes )
   {
      getSizes() = sizes;
      detail::compute_dynamic_strides< PermutationType >( getStrides(), getSizes(), getOverlaps() );
      array.setSize( getStorageSize() );
   }

   //! \brief Sets sizes of the array using raw integers.
   template< typename... IndexTypes >
   void
   setSizes( IndexTypes&&... sizes )
   {
      static_assert( sizeof...( sizes ) == getDimension(), "got wrong number of sizes" );
      detail::setSizesHelper( getSizes(), std::forward< IndexTypes >( sizes )... );
      detail::compute_dynamic_strides< PermutationType >( getStrides(), getSizes(), getOverlaps() );
      array.setSize( getStorageSize() );
   }

   /**
    * \brief Sets sizes of the array to the sizes of an existing array.
    *
    * If the array size changes, the current data will be deallocated, thus
    * all pointers and views to the array elements will become invalid.
    */
   void
   setLike( const NDArrayStorage& other )
   {
      getSizes() = other.getSizes();
      detail::compute_dynamic_strides< PermutationType >( getStrides(), getSizes(), getOverlaps() );
      array.setSize( getStorageSize() );
   }

   //! \brief Sets all elements of the array to given value.
   void
   setValue( ValueType value )
   {
      array.setValue( value );
   }

protected:
   //! \brief Underlying one-dimensional array which stores the data.
   Array array;
};

/**
 * \brief Dynamic N-dimensional array.
 *
 * \tparam Value Type of the values stored in the array.
 * \tparam SizesHolder Instance of \ref SizesHolder that will represent the
 *                     array sizes.
 * \tparam Permutation Permutation that will be applied to indices when
 *                     accessing the array elements. The identity permutation
 *                     is used by default.
 * \tparam Device Type of the \ref TNL::Devices "device" that will be used for
 *                running operations on the array.
 * \tparam Index Type of indices used for addressing the array elements.
 * \tparam Overlaps Sequence of integers representing the overlaps in each
 *                  dimension a distributed N-dimensional array.
 * \tparam Allocator Type of the allocator that will be used for allocating
 *                   elements of the array.
 *
 * See also the \ref ug_NDArrays "Users' Guide".
 */
template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Device = Devices::Host,
          typename Index = typename SizesHolder::IndexType,
          typename Overlaps = ConstStaticSizesHolder< typename SizesHolder::IndexType, SizesHolder::getDimension(), 0 >,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class NDArray
: public NDArrayStorage< Array< Value, Device, Index, Allocator >,
                         NDArrayIndexer< SizesHolder, detail::make_strides_holder< Permutation, SizesHolder >, Overlaps >,
                         Permutation >
{
   using Base =
      NDArrayStorage< Array< Value, Device, Index, Allocator >,
                      NDArrayIndexer< SizesHolder, detail::make_strides_holder< Permutation, SizesHolder >, Overlaps >,
                      Permutation >;

public:
   // inherit all constructors and assignment operators
   using Base::Base;
   using Base::operator=;

   //! \brief Constructs an empty array with zero size.
   NDArray() = default;

   //! \brief Allocator type used for allocating the array.
   using AllocatorType = Allocator;

   //! \brief Constructs an empty array and sets the provided allocator.
   NDArray( const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->array = Array< Value, Device, Index, Allocator >( allocator );
   }

   //! \brief Copy constructor with a specific allocator (makes a deep copy).
   explicit NDArray( const NDArray& other, const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->array = Array< Value, Device, Index, Allocator >( allocator );
      // copy the data
      *this = other;
   }

   //! \brief Returns the allocator associated with the array.
   [[nodiscard]] AllocatorType
   getAllocator() const
   {
      return this->array.getAllocator();
   }
};

/**
 * \brief Static N-dimensional array.
 *
 * \tparam Value Type of the values stored in the array.
 * \tparam SizesHolder Instance of \ref SizesHolder that will represent the
 *                     array sizes.
 * \tparam Permutation Permutation that will be applied to indices when
 *                     accessing the array elements. The identity permutation
 *                     is used by default.
 * \tparam Index Type of indices used for addressing the array elements.
 *
 * See also the \ref ug_NDArrays "Users' Guide".
 */
template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Index = typename SizesHolder::IndexType >
class StaticNDArray
: public NDArrayStorage< StaticArray< detail::getStaticStorageSize( SizesHolder{} ), Value >,
                         NDArrayIndexer< SizesHolder, detail::make_strides_holder< Permutation, SizesHolder > >,
                         Permutation,
                         Devices::Sequential >
{
   using Base = NDArrayStorage< StaticArray< detail::getStaticStorageSize( SizesHolder{} ), Value >,
                                NDArrayIndexer< SizesHolder, detail::make_strides_holder< Permutation, SizesHolder > >,
                                Permutation,
                                Devices::Sequential >;
   static_assert( detail::getStaticStorageSize( SizesHolder{} ) > 0, "All dimensions of a static array must be positive." );

public:
   // inherit all assignment operators
   using Base::operator=;

   // FIXME: StaticNDArray inherits even "dynamic" methods from NDArrayStorage (e.g. reset or setLike) which
   // don't make sense here - a separate base class should be created with only the static functionality

   //! \brief Sets all elements of the array to given value.
   constexpr void
   setValue( Value value )
   {
      this->array.setValue( value );
   }
};

// this is a Doxygen end-group marker
//! @}

}  // namespace TNL::Containers
