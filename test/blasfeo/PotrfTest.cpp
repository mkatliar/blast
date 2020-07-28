#include <blasfeo/Blasfeo.hpp>
#include <test/Testing.hpp>

#include <blazefeo/Blaze.hpp>


namespace blasfeo :: testing
{
    TEST(BlasfeoTest, testPotrf)
    {
        size_t const m = 5;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_C(m, m), blaze_D(m, m);
        makePositiveDefinite(blaze_C);

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_C(blaze_C), blasfeo_D(m, m);
        
        // Do syrk-potrf with BLASFEO
        potrf(m, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Check the result
        EXPECT_TRUE(isLower(blaze_blasfeo_D));
        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D * trans(blaze_blasfeo_D), blaze_C, 1e-10, 1e-10);
    }
}
