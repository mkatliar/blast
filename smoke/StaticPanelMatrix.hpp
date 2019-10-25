#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Block.hpp>
#include <smoke/PanelMatrix.hpp>

#include <blaze/util/Random.h>

#include <array>


namespace smoke
{
    template <typename T, size_t M, size_t N, size_t P = blockSize, size_t AL = blockAlignment>
    class StaticPanelMatrix
    :   public PanelMatrix<StaticPanelMatrix<T, M, N, P, AL>, P>
    {
    public:
        using ElementType = T;

        
        StaticPanelMatrix()
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            v_.fill(T {});
        }


        StaticPanelMatrix& operator=(T val)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        T operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        T& operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        size_t constexpr rows() const
        {
            return M;
        }


        size_t constexpr columns() const
        {
            return N;
        }


        size_t constexpr spacing() const
        {
            return panelColumns_ * elementsPerPanel_;
        }


        size_t constexpr panelRows() const
        {
            return panelRows_;
        }


        size_t constexpr panelColumns() const
        {
            return panelColumns_;
        }


        void pack(T const * data, size_t lda)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(T * data, size_t lda) const
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        T * block(size_t i, size_t j)
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        T const * block(size_t i, size_t j) const
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(AL) std::array<T, panelRows_ * panelColumns_ * elementsPerPanel_> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return (panel_i * panelColumns_ + panel_j) * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };
}


namespace blaze
{
    using smoke::StaticPanelMatrix;


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
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    class Rand<StaticPanelMatrix<Type, M, N, P, AL>>
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const StaticPanelMatrix<Type, M, N, P, AL> generate() const;

        template< typename Arg >
        inline const StaticPanelMatrix<Type, M, N, P, AL> generate( const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( StaticPanelMatrix<Type, M, N, P, AL>& matrix ) const;

        template< typename Arg >
        inline void randomize( StaticPanelMatrix<Type, M, N, P, AL>& matrix, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************
    };
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random StaticMatrix.
    //
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    inline const StaticPanelMatrix<Type, M, N, P, AL> Rand<StaticPanelMatrix<Type, M, N, P, AL>>::generate() const
    {
        StaticPanelMatrix<Type, M, N, P, AL> matrix;
        randomize( matrix );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random StaticMatrix.
    //
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline const StaticPanelMatrix<Type, M, N, P, AL>
    Rand<StaticPanelMatrix<Type, M, N, P, AL>>::generate( const Arg& min, const Arg& max ) const
    {
        StaticPanelMatrix<Type, M, N, P, AL> matrix;
        randomize( matrix, min, max );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a StaticMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    inline void Rand< StaticPanelMatrix<Type, M, N, P, AL> >::randomize( StaticPanelMatrix<Type, M, N, P, AL>& matrix ) const
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
    /*!\brief Randomization of a StaticMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline void Rand< StaticPanelMatrix<Type, M, N, P, AL> >::randomize( StaticPanelMatrix<Type, M, N, P, AL>& matrix,
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
    /*!\brief Setup of a random symmetric StaticMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    void makeSymmetric( StaticPanelMatrix<Type, M, N, P, AL>& matrix )
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
    /*!\brief Setup of a random symmetric StaticMatrix.
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
            , size_t P  // Panel size
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeSymmetric( StaticPanelMatrix<Type, M, N, P, AL>& matrix, const Arg& min, const Arg& max )
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
    /*!\brief Setup of a random Hermitian StaticMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    void makeHermitian( StaticPanelMatrix<Type, M, N, P, AL>& matrix )
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
    /*!\brief Setup of a random Hermitian StaticMatrix.
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
            , size_t P  // Panel size
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeHermitian( StaticPanelMatrix<Type, M, N, P, AL>& matrix, const Arg& min, const Arg& max )
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
    /*!\brief Setup of a random (Hermitian) positive definite StaticMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t M       // Number of rows
            , size_t N       // Number of columns
            , size_t P  // Panel size
            , size_t AL >      // Alignment
    void makePositiveDefinite( StaticPanelMatrix<Type, M, N, P, AL>& matrix )
    {
        using blaze::randomize;

        BLAZE_STATIC_ASSERT( M == N );
        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        randomize( matrix );
        matrix *= ctrans( matrix );

        for( size_t i=0UL; i<N; ++i ) {
            matrix(i,i) += Type(N);
        }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************
}