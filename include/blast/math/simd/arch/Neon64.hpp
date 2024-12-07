// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xsimd/xsimd.hpp>

#include <type_traits>


namespace blast
{
    namespace detail
    {
        std::size_t constexpr registerCapacity(xsimd::neon64)
        {
            return 32;
        }
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline xsimd::batch<float, Arch> maskload(float const * src, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        xsimd::batch<float, Arch> v {0.f};

        if (vgetq_lane_u32(mask, 0))
            v = vsetq_lane_f32(src[0], v, 0);

        if (vgetq_lane_u32(mask, 1))
            v = vsetq_lane_f32(src[1], v, 1);

        if (vgetq_lane_u32(mask, 2))
            v = vsetq_lane_f32(src[2], v, 2);

        if (vgetq_lane_u32(mask, 3))
            v = vsetq_lane_f32(src[3], v, 3);

        return v;
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline xsimd::batch<double, Arch> maskload(double const * src, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        xsimd::batch<double, Arch> v {0.};

        if (vgetq_lane_u64(mask, 0))
            v = vsetq_lane_f64(src[0], v, 0);

        if (vgetq_lane_u64(mask, 1))
            v = vsetq_lane_f64(src[1], v, 1);

        return v;
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline void maskstore(xsimd::batch<float, Arch> const& v, float * dst, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        if (vgetq_lane_u32(mask, 0))
            dst[0] = vgetq_lane_f32(v, 0);

        if (vgetq_lane_u32(mask, 1))
            dst[1] = vgetq_lane_f32(v, 1);

        if (vgetq_lane_u32(mask, 2))
            dst[2] = vgetq_lane_f32(v, 2);

        if (vgetq_lane_u32(mask, 3))
            dst[3] = vgetq_lane_f32(v, 3);
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline void maskstore(xsimd::batch<double, Arch> const& v, double * dst, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        if (vgetq_lane_u64(mask, 0))
            dst[0] = vgetq_lane_f64(v, 0);

        if (vgetq_lane_u64(mask, 1))
            dst[1] = vgetq_lane_f64(v, 1);
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline std::tuple<xsimd::batch<float, Arch>, xsimd::batch<std::int32_t, Arch>> imax(xsimd::batch<float, Arch> const& x, xsimd::batch<std::int32_t, Arch> const& idx) noexcept
    {
        // Step 1: Initial pairwise comparisons
        float32x4_t const y1 = vextq_f32(x, x, 1);         // Shift elements by 1: [x[1], x[2], x[3], x[0]]
        int32x4_t const iy1 = vextq_s32(idx, idx, 1);      // Shift idx by 1: [idx[1], idx[2], idx[3], idx[0]]

        uint32x4_t const mask1 = vcgtq_f32(x, y1);         // Mask for x > y1
        float32x4_t const max1 = vbslq_f32(mask1, x, y1);  // [max(x[0], x[1]), max(x[1], x[2]), max(x[2], x[3]), max(x[3], x[0])]
        int32x4_t const idx1 = vbslq_s32(mask1, idx, iy1); // Blend idx and iy1 based on mask

        // Step 2: Second pairwise comparison on the result from Step 1
        float32x4_t const y2 = vextq_f32(max1, max1, 2);   // Shift elements by 2: [max1[2], max1[3], max1[0], max1[1]]
        int32x4_t const iy2 = vextq_s32(idx1, idx1, 2);    // Shift idx1 by 2: [idx1[2], idx1[3], idx1[0], idx1[1]]

        uint32x4_t const mask2 = vcgtq_f32(max1, y2);      // Mask for max1 > y2
        float32x4_t const max2 = vbslq_f32(mask2, max1, y2); // Blend max1 and y2 based on mask
        int32x4_t const idx2 = vbslq_s32(mask2, idx1, iy2);  // Blend idx1 and iy2 based on mask

        // Return the max value and corresponding index, which are now the same across all lanes
        return {max2, idx2};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::neon64, Arch>
    inline std::tuple<xsimd::batch<double, Arch>, xsimd::batch<std::int64_t, Arch>> imax(xsimd::batch<double, Arch> const& x, xsimd::batch<std::int64_t, Arch> const& idx) noexcept
    {
        // Swap elements of x
        float64x2_t const y = vextq_f64(x, x, 1);
        int64x2_t const iy = vextq_s64(idx, idx, 1);

        // Compare
        uint64x2_t const mask = vcgtq_f64(x, y);

        // Blend
        float64x2_t const m = vbslq_f64(mask, x, y);
        int64x2_t const im = vbslq_s64(mask, idx, iy);

        return {m, im};
    }
}
