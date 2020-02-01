#pragma once

#include <blasfeo/Alloc.hpp>

#include <algorithm>


namespace blasfeo
{
    namespace detail
    {
        /// @brief Aligned storage for BLASFEO matricx and vector data arrays.
        ///
        /// Behaves similarly to std::uniqie_ptr<>. Uses BLASFEO functions to allocate/deallocate memory.
        class AlignedStorage
        {
        public:
            AlignedStorage()
            {                
            }

            
            explicit AlignedStorage(size_t bytes)
            :   ptr_(malloc_align(bytes))
            {
            }


            /// @brief Copy ctor deleted
            AlignedStorage(AlignedStorage const&) = delete;


            /// @brief Move ctor
            AlignedStorage(AlignedStorage&& rhs) noexcept
            :   ptr_(rhs.ptr_)
            {
                rhs.ptr_ = nullptr;
            }


            /// @brief Copy-assignment deleted
            AlignedStorage& operator=(AlignedStorage const&) = delete;


            /// @brief Move assignment
            AlignedStorage& operator=(AlignedStorage&& rhs) noexcept
            {
                swap(rhs);

                return *this;
            }


            /// @brief Swap with another AlignedStorage
            void swap(AlignedStorage& rhs) noexcept
            {
                std::swap(ptr_, rhs.ptr_);
            }


            /// @brief Get data pointer
            void * get() noexcept
            {
                return ptr_;
            }


            /// @brief Get const data pointer
            void const * get() const noexcept
            {
                return ptr_;
            }


            ~AlignedStorage()
            {
                free_align(ptr_);
            }


        private:
            void * ptr_ = nullptr;
        };
    }
}