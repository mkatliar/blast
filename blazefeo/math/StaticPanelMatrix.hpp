#pragma once

#include <blazefeo/math/panel/StaticPanelMatrix.hpp>

#include <blaze/util/Random.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>


namespace blaze
{
    using blazefeo::StaticPanelMatrix;


    //=================================================================================================
    //
    //  RAND SPECIALIZATION
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the Rand class template for StaticPanelMatrix.
    // \ingroup random
    //
    // This specialization of the Rand class creates random instances of StaticPanelMatrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    class Rand<StaticPanelMatrix<Type, M, N, SO>>
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const StaticPanelMatrix<Type, M, N, SO> generate() const;

        template< typename Arg >
        inline const StaticPanelMatrix<Type, M, N, SO> generate( const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( StaticPanelMatrix<Type, M, N, SO>& matrix ) const;

        template< typename Arg >
        inline void randomize( StaticPanelMatrix<Type, M, N, SO>& matrix, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************
    };
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random StaticPanelMatrix.
    //
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    inline const StaticPanelMatrix<Type, M, N, SO> Rand<StaticPanelMatrix<Type, M, N, SO>>::generate() const
    {
        StaticPanelMatrix<Type, M, N, SO> matrix;
        randomize( matrix );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random StaticPanelMatrix.
    //
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    template< typename Arg >  // Min/max argument type
    inline const StaticPanelMatrix<Type, M, N, SO>
    Rand<StaticPanelMatrix<Type, M, N, SO>>::generate( const Arg& min, const Arg& max ) const
    {
        StaticPanelMatrix<Type, M, N, SO> matrix;
        randomize( matrix, min, max );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    inline void Rand< StaticPanelMatrix<Type, M, N, SO> >::randomize( StaticPanelMatrix<Type, M, N, SO>& matrix ) const
    {
        using blaze::randomize;

        for( size_t i=0UL; i<M; ++i ) {
            for( size_t j=0UL; j<N; ++j ) {
                randomize( matrix(i,j) );
            }
        }
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    template< typename Arg >  // Min/max argument type
    inline void Rand< StaticPanelMatrix<Type, M, N, SO> >::randomize( StaticPanelMatrix<Type, M, N, SO>& matrix,
                                                            const Arg& min, const Arg& max ) const
    {
        using blaze::randomize;

        for( size_t i=0UL; i<M; ++i ) {
            for( size_t j=0UL; j<N; ++j ) {
                randomize( matrix(i,j), min, max );
            }
        }
    }
    /*! \endcond */
    //*************************************************************************************************




    //=================================================================================================
    //
    //  MAKE FUNCTIONS
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Setup of a random symmetric StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    void makeSymmetric( StaticPanelMatrix<Type, M, N, SO>& matrix )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );

        for( size_t i=0UL; i<N; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
                randomize( matrix(i,j) );
                matrix(j,i) = matrix(i,j);
            }
            randomize( matrix(i,i) );
        }

        BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Setup of a random symmetric StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO
            , typename Arg >  // Min/max argument type
    void makeSymmetric( StaticPanelMatrix<Type, M, N, SO>& matrix, const Arg& min, const Arg& max )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );

        for( size_t i=0UL; i<N; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
                randomize( matrix(i,j), min, max );
                matrix(j,i) = matrix(i,j);
            }
            randomize( matrix(i,i), min, max );
        }

        BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Setup of a random Hermitian StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    void makeHermitian( StaticPanelMatrix<Type, M, N, SO>& matrix )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );
        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        using BT = UnderlyingBuiltin_t<Type>;

        for( size_t i=0UL; i<N; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
                randomize( matrix(i,j) );
                matrix(j,i) = conj( matrix(i,j) );
            }
            matrix(i,i) = rand<BT>();
        }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-Hermitian matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Setup of a random Hermitian StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO
            , typename Arg >  // Min/max argument type
    void makeHermitian( StaticPanelMatrix<Type, M, N, SO>& matrix, const Arg& min, const Arg& max )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );
        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        using BT = UnderlyingBuiltin_t<Type>;

        for( size_t i=0UL; i<N; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
                randomize( matrix(i,j), min, max );
                matrix(j,i) = conj( matrix(i,j) );
            }
            matrix(i,i) = rand<BT>( real( min ), real( max ) );
        }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-Hermitian matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Setup of a random (Hermitian) positive definite StaticPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , bool SO >             
    void makePositiveDefinite( StaticPanelMatrix<Type, M, N, SO>& matrix )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );
        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        matrix = Type {};

        for (size_t i = 0; i < N; ++i)
            matrix(i, i) = Type(N);

        StaticPanelMatrix<Type, N, N, SO> A;
        randomize(A);

        gemm_nt(A, A, matrix, matrix);

        // TODO: implement it as below after the matrix *= ctrans( matrix ) expression works.
        
        // randomize( matrix );
        // matrix *= ctrans( matrix );

        // for( size_t i=0UL; i<N; ++i ) {
        //     matrix(i,i) += Type(N);
        // }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************
}