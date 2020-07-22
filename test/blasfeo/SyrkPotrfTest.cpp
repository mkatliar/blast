#include <blasfeo/Blasfeo.hpp>
#include <test/Testing.hpp>

#include <blaze/Math.h>


namespace blasfeo :: testing
{
    TEST(BlasfeoTest, testSyrkPotrf)
    {
        size_t const m = 5, k = 4;  // <-- Sic!

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k), blaze_C(m, m), blaze_D(m, m);
        randomize(blaze_A);
        makePositiveDefinite(blaze_C);

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_A(blaze_A), blasfeo_C(blaze_C), blasfeo_D(m, m);
        
        // Do syrk-potrf with BLASFEO
        syrk_potrf(m, k, blasfeo_A, 0, 0, blasfeo_A, 0, 0, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Check the result
        EXPECT_TRUE(isLower(blaze_blasfeo_D));
        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D * trans(blaze_blasfeo_D), blaze_C + blaze_A * trans(blaze_A), 1e-10, 1e-10);
    }
}
