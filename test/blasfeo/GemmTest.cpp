// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blasfeo/Blasfeo.hpp>
#include <test/Testing.hpp>

#include <blazefeo/Blaze.hpp>


namespace blasfeo :: testing
{
    TEST(BlasfeoTest, testGemmTN)
    {
        size_t const m = 5, n = 6, k = 4;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(k, m), blaze_B(k, n), blaze_C(m, n), blaze_D(m, n);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Do gemm with Blaze
        //
        blaze_D = blaze_C + trans(blaze_A) * blaze_B;
        // std::cout << "blaze_D=\n" << blaze_D;

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_A(blaze_A), blasfeo_B(blaze_B), blasfeo_C(blaze_C), blasfeo_D(m, n);
        
        // Do gemm with BLASFEO
        gemm_tn(m, n, k, 1.0, blasfeo_A, 0, 0, blasfeo_B, 0, 0, 1.0, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D, blaze_D, 1e-10, 1e-10);
    }


    TEST(BlasfeoTest, testGemmNN)
    {
        size_t const m = 5, n = 6, k = 4;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k), blaze_B(k, n), blaze_C(m, n), blaze_D(m, n);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Do gemm with Blaze
        //
        blaze_D = blaze_C + blaze_A * blaze_B;
        // std::cout << "blaze_D=\n" << blaze_D;

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_A(blaze_A), blasfeo_B(blaze_B), blasfeo_C(blaze_C), blasfeo_D(m, n);
        
        // Do gemm with BLASFEO
        gemm_nn(m, n, k, 1.0, blasfeo_A, 0, 0, blasfeo_B, 0, 0, 1.0, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D, blaze_D, 1e-10, 1e-10);
    }


    TEST(BlasfeoTest, testGemmNT)
    {
        size_t const m = 5, n = 6, k = 4;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k), blaze_B(n, k), blaze_C(m, n), blaze_D(m, n);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Do gemm with Blaze
        //
        blaze_D = blaze_C + blaze_A * trans(blaze_B);
        // std::cout << "blaze_D=\n" << blaze_D;

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_A(blaze_A), blasfeo_B(blaze_B), blasfeo_C(blaze_C), blasfeo_D(m, n);
        
        // Do gemm with BLASFEO
        gemm_nt(m, n, k, 1.0, blasfeo_A, 0, 0, blasfeo_B, 0, 0, 1.0, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D, blaze_D, 1e-10, 1e-10);
    }
}
