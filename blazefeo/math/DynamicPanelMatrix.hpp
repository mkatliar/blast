#pragma once

#include <blazefeo/math/panel/DynamicPanelMatrix.hpp>

#include <blaze/util/Random.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>


namespace blaze
{
    using blazefeo::DynamicPanelMatrix;


    //=================================================================================================
    //
    //  RAND SPECIALIZATION
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the Rand class template for DynamicPanelMatrix.
    // \ingroup random
    //
    // This specialization of the Rand class creates random instances of DynamicPanelMatrix.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    class Rand< DynamicPanelMatrix<Type, SO> >
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const DynamicPanelMatrix<Type, SO> generate( size_t m, size_t n ) const;

        template< typename Arg >
        inline const DynamicPanelMatrix<Type, SO> generate( size_t m, size_t n, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( DynamicPanelMatrix<Type, SO>& matrix ) const;

        template< typename Arg >
        inline void randomize( DynamicPanelMatrix<Type, SO>& matrix, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************
    };
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random DynamicPanelMatrix.
    //
    // \param m The number of rows of the random matrix.
    // \param n The number of columns of the random matrix.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    inline const DynamicPanelMatrix<Type, SO>
    Rand< DynamicPanelMatrix<Type, SO> >::generate( size_t m, size_t n ) const
    {
        DynamicPanelMatrix<Type, SO> matrix( m, n );
        randomize( matrix );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random DynamicPanelMatrix.
    //
    // \param m The number of rows of the random matrix.
    // \param n The number of columns of the random matrix.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    template< typename Arg >  // Min/max argument type
    inline const DynamicPanelMatrix<Type, SO>
    Rand< DynamicPanelMatrix<Type, SO> >::generate( size_t m, size_t n, const Arg& min, const Arg& max ) const
    {
        DynamicPanelMatrix<Type, SO> matrix( m, n );
        randomize( matrix, min, max );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    inline void Rand< DynamicPanelMatrix<Type, SO> >::randomize( DynamicPanelMatrix<Type, SO>& matrix ) const
    {
        using blaze::randomize;

        const size_t m( matrix.rows()    );
        const size_t n( matrix.columns() );

        for( size_t i=0UL; i<m; ++i ) {
            for( size_t j=0UL; j<n; ++j ) {
                randomize( matrix(i,j) );
            }
        }
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    template< typename Arg >  // Min/max argument type
    inline void Rand< DynamicPanelMatrix<Type, SO> >::randomize( DynamicPanelMatrix<Type, SO>& matrix,
                                                        const Arg& min, const Arg& max ) const
    {
        using blaze::randomize;

        const size_t m( matrix.rows()    );
        const size_t n( matrix.columns() );

        for( size_t i=0UL; i<m; ++i ) {
            for( size_t j=0UL; j<n; ++j ) {
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
    /*!\brief Setup of a random symmetric DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    void makeSymmetric( DynamicPanelMatrix<Type, SO>& matrix )
    {
        using blaze::randomize;

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        for( size_t i=0UL; i<n; ++i ) {
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
    /*!\brief Setup of a random symmetric DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , bool SO
            , typename Arg >  // Min/max argument type
    void makeSymmetric( DynamicPanelMatrix<Type, SO>& matrix, const Arg& min, const Arg& max )
    {
        using blaze::randomize;

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        for( size_t i=0UL; i<n; ++i ) {
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
    /*!\brief Setup of a random Hermitian DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    void makeHermitian( DynamicPanelMatrix<Type, SO>& matrix )
    {
        using blaze::randomize;

        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        using BT = UnderlyingBuiltin_t<Type>;

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        for( size_t i=0UL; i<n; ++i ) {
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
    /*!\brief Setup of a random Hermitian DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , bool SO
            , typename Arg >  // Min/max argument type
    void makeHermitian( DynamicPanelMatrix<Type, SO>& matrix, const Arg& min, const Arg& max )
    {
        using blaze::randomize;

        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        using BT = UnderlyingBuiltin_t<Type>;

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        for( size_t i=0UL; i<n; ++i ) {
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
    /*!\brief Setup of a random (Hermitian) positive definite DynamicPanelMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , bool SO >
    void makePositiveDefinite( DynamicPanelMatrix<Type, SO>& matrix )
    {
        using blaze::randomize;

        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        matrix = Type {};

        for (size_t i = 0; i < n; ++i)
            matrix(i, i) = Type(n);

        DynamicPanelMatrix<Type, SO> A(n, n);
        randomize(A);

        gemm_nt(A, A, matrix, matrix);

        // TODO: implement it as below after the matrix *= ctrans( matrix ) expression works.

        // randomize( matrix );
        // matrix *= ctrans( matrix );

        // for( size_t i=0UL; i<n; ++i ) {
        //     matrix(i,i) += Type(n);
        // }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************

}