#pragma once

// Intel compiler
#if defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)
#   define BLAST_RESTRICT __restrict

// GNU compiler
#elif defined(__GNUC__)
#   define BLAST_RESTRICT __restrict

// Microsoft visual studio
#elif defined(_MSC_VER)
#   define BLAST_RESTRICT __restrict

// All other compilers
#else
#   define BLAST_RESTRICT
#  endif
