// Copyright 2024 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <xsimd/xsimd.hpp>

#include <type_traits>


namespace blast
{
    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline xsimd::batch<float, Arch> maskload(float const * src, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        return _mm256_maskload_ps(src, _mm256_castps_si256(mask));
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline xsimd::batch<double, Arch> maskload(double const * src, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        return _mm256_maskload_pd(src, _mm256_castpd_si256(mask));
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline void maskstore(xsimd::batch<float, Arch> const& v, float * dst, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        _mm256_maskstore_ps(dst, _mm256_castps_si256(mask), v);
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline void maskstore(xsimd::batch<double, Arch> const& v, double * dst, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        _mm256_maskstore_pd(dst, _mm256_castpd_si256(mask), v);
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline std::tuple<xsimd::batch<float, Arch>, xsimd::batch<std::int32_t, Arch>> imax(xsimd::batch<float, Arch> const& v1, xsimd::batch<std::int32_t, Arch> const& idx) noexcept
    {
        /* v2 = [G H E F | C D A B]                                                                         */
        __m256 const v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
        __m256i const iv2 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(idx), 0b10'11'00'01));

        /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
        /* v3 = [W W Z Z | Y Y X X]                                                                         */
        // __m256 v3 = _mm256_max_ps(v1, v2);
        __m256 const mask_v3 = _mm256_cmp_ps(v2, v1, _CMP_GT_OQ);
        __m256 const v3 = _mm256_blendv_ps(v1, v2, mask_v3);
        __m256i const iv3 = _mm256_blendv_epi8(idx, iv2, _mm256_castps_si256(mask_v3));

        /* v4 = [Z Z W W | X X Y Y]                                                                         */
        __m256 const v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
        __m256i const iv4 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(iv3), 0b00'00'10'10));

        /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
        /* v5 = [J J J J | I I I I]                                                                         */
        // __m256 v5 = _mm256_max_ps(v3, v4);
        __m256 const mask_v5 = _mm256_cmp_ps(v4, v3, _CMP_GT_OQ);
        __m256 const v5 = _mm256_blendv_ps(v3, v4, mask_v5);
        __m256i const iv5 = _mm256_blendv_epi8(iv3, iv4, _mm256_castps_si256(mask_v5));

        /* v6 = [I I I I | J J J J]                                                                         */
        __m256 const v6 = _mm256_permute2f128_ps(v5, v5, 0b0000'0001);
        __m256i const iv6 = _mm256_castps_si256(
            _mm256_permute2f128_ps(
                _mm256_castsi256_ps(iv5),
                _mm256_castsi256_ps(iv5),
                0b0000'0001
            )
        );

        /* v7 = [M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J) | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)] */
        // __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);
        __m256 const mask_v7 = _mm256_cmp_ps(v6, v5, _CMP_GT_OQ);
        __m256 const v7 = _mm256_blendv_ps(v5, v6, mask_v7);
        __m256i const iv7 = _mm256_blendv_epi8(iv5, iv6, _mm256_castps_si256(mask_v7));

        return {v7, iv7};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline std::tuple<xsimd::batch<double, Arch>, xsimd::batch<std::int64_t, Arch>> imax(xsimd::batch<double, Arch> const& x, xsimd::batch<std::int64_t, Arch> const& idx) noexcept
    {
        __m256d const y = _mm256_permute2f128_pd(x, x, 1); // permute 128-bit values
        __m256i const iy = _mm256_permute2f128_si256(idx, idx, 1);

        // __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        __m256d const mask_m1 = _mm256_cmp_pd(y, x, _CMP_GT_OQ);
        __m256d const m1 = _mm256_blendv_pd(x, y, mask_m1);
        __m256i const im1 = _mm256_blendv_epi8(idx, iy, _mm256_castpd_si256(mask_m1));

        __m256d const m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
        __m256i const im2 = _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(im1), 5));

        // __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
        __m256d const mask_m = _mm256_cmp_pd(m2, m1, _CMP_GT_OQ);
        __m256d const m = _mm256_blendv_pd(m1, m2, mask_m);
        __m256i const im = _mm256_blendv_epi8(im1, im2, _mm256_castpd_si256(mask_m));

        return {m, im};
    }
}
