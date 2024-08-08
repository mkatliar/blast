// Copyright 2023-2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/dense/DynamicVectorPointer.hpp>
#include <blast/math/dense/StaticVectorPointer.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/system/Inline.hpp>


namespace blast
{
    /**
     * @brief Pointer to the first element of a vector
     *
     * @tparam VT vector type
     * @param v vector
     *
     * @return pointer to the first element of @a v
     */
    template <typename VT>
    requires IsDenseVector_v<VT>
    BLAST_ALWAYS_INLINE auto ptr(VT& v)
    {
        return ptr<IsAligned_v<VT>>(v, 0);
    }


    /**
     * @brief Pointer to the first element of a const vector
     *
     * @tparam VT vector type
     * @param v vector
     *
     * @return pointer to the first element of @a v
     */
    template <typename VT>
    requires IsDenseVector_v<VT>
    BLAST_ALWAYS_INLINE auto ptr(VT const& v)
    {
        return ptr<IsAligned_v<VT>>(v, 0);
    }
}
