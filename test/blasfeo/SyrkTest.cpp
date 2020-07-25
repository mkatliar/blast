#include <blasfeo/Blasfeo.hpp>
#include <test/Testing.hpp>

#include <blaze/Math.h>


namespace blasfeo :: testing
{
    TEST(BlasfeoTest, testSyrkLn)
    {
        size_t const m = 5, k = 4;  // <-- Sic!

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k), blaze_B(m, k);
        blaze::LowerMatrix<blaze::DynamicMatrix<double, blaze::columnMajor>> blaze_C(m, m);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> A(blaze_A), B(blaze_B), C(blaze_C), D(m, m);
        
        // Do syrk with BLASFEO
        syrk_ln(m, k, 1., A, 0, 0, B, 0, 0, 1., C, 0, 0, D, 0, 0);

        // Calculate the correct result
        auto const D_ref = evaluate(blaze_C + blaze_A * trans(blaze_B));

        // Check the result
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j <= i; ++j)
                EXPECT_NEAR(D(i, j), D_ref(i, j), 1e-10);
            
            for (size_t j = i + 1; j <= m; ++j)
                EXPECT_EQ(D(i, j), 0.);
        }
    }
}
