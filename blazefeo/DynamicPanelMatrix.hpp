#pragma once

#include <blazefeo/SizeT.hpp>
#include <blazefeo/Block.hpp>
#include <blazefeo/PanelMatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Random.h>
#include <blaze/math/shims/NextMultiple.h>

#include <memory>
#include <cstdlib>
#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <typename Type, bool SO = rowMajor, size_t AL = blockAlignment>
    class DynamicPanelMatrix
    :   public PanelMatrix<DynamicPanelMatrix<Type, SO, AL>, SO>
    {
    public:
        using ElementType = Type;

        
        DynamicPanelMatrix(size_t m, size_t n)
        :   m_(m)
        ,   n_(n)
        ,   spacing_(TILE_SIZE * nextMultiple(n, TILE_SIZE))
        ,   capacity_(nextMultiple(m, TILE_SIZE) * nextMultiple(n, TILE_SIZE))
        ,   v_(reinterpret_cast<Type *>(std::aligned_alloc(AL, capacity_ * sizeof(Type))), &std::free)
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_.get(), capacity_, Type {});
        }


        DynamicPanelMatrix& operator=(Type val)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        Type operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        Type& operator()(size_t i, size_t j)
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


        void pack(Type const * data, size_t lda)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(Type * data, size_t lda) const
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        Type * block(size_t i, size_t j)
        {
            return v_.get() + i * spacing_ + j * ELEMENTS_PER_TILE;
        }


        Type const * block(size_t i, size_t j) const
        {
            return v_.get() + i * spacing_ + j * ELEMENTS_PER_TILE;
        }


    private:
        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;
        
        std::unique_ptr<Type[], decltype(&std::free)> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / TILE_SIZE;
            size_t const panel_j = j / TILE_SIZE;
            size_t const subpanel_i = i % TILE_SIZE;
            size_t const subpanel_j = j % TILE_SIZE;

            return panel_i * spacing_ + panel_j * ELEMENTS_PER_TILE + subpanel_i + subpanel_j * TILE_SIZE;
        }
    };
}


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
            , bool SO
            , size_t AL >      // Alignment
    class Rand< DynamicPanelMatrix<Type, SO, AL> >
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const DynamicPanelMatrix<Type, SO, AL> generate( size_t m, size_t n ) const;

        template< typename Arg >
        inline const DynamicPanelMatrix<Type, SO, AL> generate( size_t m, size_t n, const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( DynamicPanelMatrix<Type, SO, AL>& matrix ) const;

        template< typename Arg >
        inline void randomize( DynamicPanelMatrix<Type, SO, AL>& matrix, const Arg& min, const Arg& max ) const;
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
            , bool SO
            , size_t AL >      // Alignment
    inline const DynamicPanelMatrix<Type, SO, AL>
    Rand< DynamicPanelMatrix<Type, SO, AL> >::generate( size_t m, size_t n ) const
    {
        DynamicPanelMatrix<Type, SO, AL> matrix( m, n );
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
            , bool SO
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline const DynamicPanelMatrix<Type, SO, AL>
    Rand< DynamicPanelMatrix<Type, SO, AL> >::generate( size_t m, size_t n, const Arg& min, const Arg& max ) const
    {
        DynamicPanelMatrix<Type, SO, AL> matrix( m, n );
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
            , bool SO
            , size_t AL >      // Alignment
    inline void Rand< DynamicPanelMatrix<Type, SO, AL> >::randomize( DynamicPanelMatrix<Type, SO, AL>& matrix ) const
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
            , bool SO
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline void Rand< DynamicPanelMatrix<Type, SO, AL> >::randomize( DynamicPanelMatrix<Type, SO, AL>& matrix,
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
            , bool SO
            , size_t AL >      // Alignment
    void makeSymmetric( DynamicPanelMatrix<Type, SO, AL>& matrix )
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
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeSymmetric( DynamicPanelMatrix<Type, SO, AL>& matrix, const Arg& min, const Arg& max )
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
            , bool SO
            , size_t AL >      // Alignment
    void makeHermitian( DynamicPanelMatrix<Type, SO, AL>& matrix )
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
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeHermitian( DynamicPanelMatrix<Type, SO, AL>& matrix, const Arg& min, const Arg& max )
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
            , bool SO
            , size_t AL >      // Alignment
    void makePositiveDefinite( DynamicPanelMatrix<Type, SO, AL>& matrix )
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