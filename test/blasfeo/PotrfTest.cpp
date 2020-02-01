#include <blasfeo/Blasfeo.hpp>
#include <test/Testing.hpp>

#include <blaze/Math.h>


namespace blasfeo :: testing
{
    TEST(BlasfeoTest, testPotrf)
    {
        size_t const m = 5;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_C(m, m), blaze_D(m, m);
        makePositiveDefinite(blaze_C);

        // Do potrf with Blaze
        //
        llh(blaze_C, blaze_D);
        // std::cout << "blaze_D=\n" << blaze_D;

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_C(blaze_C), blasfeo_D(m, m);
        
        // Do syrk-potrf with BLASFEO
        potrf(m, blasfeo_C, 0, 0, blasfeo_D, 0, 0);

        // Copy the resulting D matrix from BLASFEO to Blaze
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_blasfeo_D;
        blasfeo_D.unpack(blaze_blasfeo_D);

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_blasfeo_D, blaze_D, 1e-10, 1e-10);
    }
}
