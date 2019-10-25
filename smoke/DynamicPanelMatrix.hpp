#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Block.hpp>
#include <smoke/PaddedSize.hpp>
#include <smoke/PanelMatrix.hpp>

#include <blaze/util/Random.h>

#include <memory>
#include <cstdlib>
#include <algorithm>


namespace smoke
{
    template <typename T, size_t P = blockSize, size_t AL = blockAlignment>
    class DynamicPanelMatrix
    :   public PanelMatrix<DynamicPanelMatrix<T, P, AL>, P>
    {
    public:
        using ElementType = T;

        
        DynamicPanelMatrix(size_t m, size_t n)
        :   m_(m)
        ,   n_(n)
        ,   spacing_(P * paddedSize(n, P))
        ,   capacity_(paddedSize(m, P) * paddedSize(n, P))
        ,   v_(reinterpret_cast<T *>(std::aligned_alloc(AL, capacity_ * sizeof(T))), &std::free)
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_.get(), capacity_, T {});
        }


        DynamicPanelMatrix& operator=(T val)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
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


        size_t rows() const
        {
            return m_;
        }


        size_t columns() const
        {
            return n_;
        }


        size_t spacing() const
        {
            return spacing_;
        }


        void pack(T const * data, size_t lda)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(T * data, size_t lda) const
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        T * block(size_t i, size_t j)
        {
            return v_.get() + i * spacing_ + j * elementsPerPanel_;
        }


        T const * block(size_t i, size_t j) const
        {
            return v_.get() + i * spacing_ + j * elementsPerPanel_;
        }


    private:
        static size_t constexpr elementsPerPanel_ = P * P;

        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;
        
        std::unique_ptr<T[], decltype(&std::free)> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return panel_i * spacing_ + panel_j * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };
}


namespace blaze
{
    using smoke::DynamicPanelMatrix;


    //=================================================================================================
    //
    //  RAND SPECIALIZATION
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the Rand class template for DynamicMatrix.
    // \ingroup random
    //
    // This specialization of the Rand class creates random instances of DynamicMatrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    class Rand< DynamicPanelMatrix<Type, P, AL> >
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const DynamicPanelMatrix<Type, P, AL> generate( size_t m, size_t n ) const;

        template< typename Arg >
        inline const DynamicPanelMatrix<Type, P, AL> generate( size_t m, size_t n, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( DynamicPanelMatrix<Type, P, AL>& matrix ) const;

        template< typename Arg >
        inline void randomize( DynamicPanelMatrix<Type, P, AL>& matrix, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************
    };
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random DynamicMatrix.
    //
    // \param m The number of rows of the random matrix.
    // \param n The number of columns of the random matrix.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    inline const DynamicPanelMatrix<Type, P, AL>
    Rand< DynamicPanelMatrix<Type, P, AL> >::generate( size_t m, size_t n ) const
    {
        DynamicPanelMatrix<Type, P, AL> matrix( m, n );
        randomize( matrix );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Generation of a random DynamicMatrix.
    //
    // \param m The number of rows of the random matrix.
    // \param n The number of columns of the random matrix.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return The generated random matrix.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline const DynamicPanelMatrix<Type, P, AL>
    Rand< DynamicPanelMatrix<Type, P, AL> >::generate( size_t m, size_t n, const Arg& min, const Arg& max ) const
    {
        DynamicPanelMatrix<Type, P, AL> matrix( m, n );
        randomize( matrix, min, max );
        return matrix;
    }
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Randomization of a DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    inline void Rand< DynamicPanelMatrix<Type, P, AL> >::randomize( DynamicPanelMatrix<Type, P, AL>& matrix ) const
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
    /*!\brief Randomization of a DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline void Rand< DynamicPanelMatrix<Type, P, AL> >::randomize( DynamicPanelMatrix<Type, P, AL>& matrix,
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
    /*!\brief Setup of a random symmetric DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    void makeSymmetric( DynamicPanelMatrix<Type, P, AL>& matrix )
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
    /*!\brief Setup of a random symmetric DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeSymmetric( DynamicPanelMatrix<Type, P, AL>& matrix, const Arg& min, const Arg& max )
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
    /*!\brief Setup of a random Hermitian DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    void makeHermitian( DynamicPanelMatrix<Type, P, AL>& matrix )
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
    /*!\brief Setup of a random Hermitian DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \param min The smallest possible value for a matrix element.
    // \param max The largest possible value for a matrix element.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeHermitian( DynamicPanelMatrix<Type, P, AL>& matrix, const Arg& min, const Arg& max )
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
    /*!\brief Setup of a random (Hermitian) positive definite DynamicMatrix.
    //
    // \param matrix The matrix to be randomized.
    // \return void
    // \exception std::invalid_argument Invalid non-square matrix provided.
    */
    template< typename Type  // Data type of the matrix
            , size_t P  // Block size
            , size_t AL >      // Alignment
    void makePositiveDefinite( DynamicPanelMatrix<Type, P, AL>& matrix )
    {
        using blaze::randomize;

        BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

        if( !isSquare( ~matrix ) ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
        }

        const size_t n( matrix.rows() );

        randomize( matrix );
        matrix *= ctrans( matrix );

        for( size_t i=0UL; i<n; ++i ) {
            matrix(i,i) += Type(n);
        }

        BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-symmetric matrix detected" );
    }
    /*! \endcond */
    //*************************************************************************************************
}