// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#   define BLAST_STRONG_INLINE __forceinline
#else
#   define BLAST_STRONG_INLINE inline
#endif

#if defined(__GNUC__)
#   define BLAST_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#   define BLAST_ALWAYS_INLINE BLAST_STRONG_INLINE
#endif
