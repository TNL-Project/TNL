// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "Functions.h"

namespace TNL::Backend {

class Stream
{
private:
   struct Wrapper
   {
      stream_t handle = 0;  // NOLINT(modernize-use-nullptr)

      Wrapper() = default;
      Wrapper( const Wrapper& other ) = delete;
      Wrapper( Wrapper&& other ) noexcept = default;
      Wrapper&
      operator=( const Wrapper& other ) = delete;
      Wrapper&
      operator=( Wrapper&& other ) noexcept = default;

      Wrapper( stream_t handle )
      : handle( handle )
      {}

      ~Wrapper()
      {
         streamDestroy( handle );
      }
   };

   std::shared_ptr< Wrapper > wrapper;

   //! \brief Internal constructor for the factory methods - initialization by the wrapper.
   Stream( std::shared_ptr< Wrapper >&& wrapper )
   : wrapper( std::move( wrapper ) )
   {}

public:
   //! \brief Constructs a stream wrapping the CUDA 0 (`NULL`) stream.
   Stream() = default;

   //! \brief Default copy-constructor.
   Stream( const Stream& other ) = default;

   //! \brief Default move-constructor.
   Stream( Stream&& other ) noexcept = default;

   //! \brief Default copy-assignment operator.
   Stream&
   operator=( const Stream& other ) = default;

   //! \brief Default move-assignment operator.
   Stream&
   operator=( Stream&& other ) noexcept = default;

   /**
    * \brief Creates a new stream.
    *
    * The stream is created by calling \e cudaStreamCreateWithPriority with the
    * following parameters:
    *
    * \param flags Custom flags for stream creation. Possible values are:
    *    - \e cudaStreamDefault: Default stream creation flag.
    *    - \e cudaStreamNonBlocking: Specifies that work running in the created
    *      stream may run concurrently with work in stream 0 (the `NULL`
    *      stream), and that the created stream should perform no implicit
    *      synchronization with stream 0.
    * \param priority Priority of the stream. Lower numbers represent higher
    *    priorities. See \e cudaDeviceGetStreamPriorityRange for more
    *    information about the meaningful stream priorities that can be passed.
    */
   static Stream
   create( unsigned int flags = StreamDefault, int priority = 0 )
   {
      return { std::make_shared< Wrapper >( streamCreateWithPriority( flags, priority ) ) };
   }

   /**
    * \brief Access the CUDA stream handle associated with this object.
    *
    * This routine permits the implicit conversion from \ref Stream to
    * `stream_t`.
    *
    * \warning The obtained `stream_t` handle becomes invalid when the
    * originating \ref Stream object is destroyed. For example, the following
    * code is invalid, because the \ref Stream object managing the lifetime of
    * the `stream_t` handle is destroyed as soon as it is cast to `stream_t`:
    *
    * \code{.cpp}
    * const TNL::Backend::stream_t stream = TNL::Backend::Stream::create();
    * my_kernel<<< gridSize, blockSize, 0, stream >>>( args... );
    * \endcode
    */
   operator const stream_t&() const
   {
      return wrapper->handle;
   }
};

}  // namespace TNL::Backend
