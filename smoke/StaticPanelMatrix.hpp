#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Block.hpp>
#include <smoke/PanelMatrix.hpp>
#include <smoke/Gemm.hpp>

#include <blaze/util/Random.h>

#include <array>


namespace smoke
{
    using namespace blaze;


    template <typename Type, size_t M, size_t N, size_t P = blockSize, bool SO = rowMajor, size_t AL = blockAlignment>
    class StaticPanelMatrix
    :   public PanelMatrix<StaticPanelMatrix<Type, M, N, P, SO, AL>, P, SO>
    {
    public:
        //**Type definitions****************************************************************************
        using This          = StaticPanelMatrix<Type, M, N, P, SO, AL>;   //!< Type of this StaticPanelMatrix instance.
        using BaseType      = PanelMatrix<This, P>;        //!< Base type of this StaticPanelMatrix instance.
        using ResultType    = This;                        //!< Result type for expression template evaluations.
        using OppositeType  = StaticPanelMatrix<Type, M, N, P, !SO, AL>;  //!< Result type with opposite storage order for expression template evaluations.
        using TransposeType = StaticPanelMatrix<Type, N, M, P, !SO, AL>;  //!< Transpose type for expression template evaluations.
        using ElementType   = Type;                        //!< Type of the matrix elements.
        // using SIMDType      = SIMDTrait_t<ElementType>;    //!< SIMD type of the matrix elements.
        using ReturnType    = const Type&;                 //!< Return type for expression template evaluations.
        using CompositeType = const This&;                 //!< Data type for composite expression templates.

        using Reference      = Type&;        //!< Reference to a non-constant matrix value.
        using ConstReference = const Type&;  //!< Reference to a constant matrix value.
        using Pointer        = Type*;        //!< Pointer to a non-constant matrix value.
        using ConstPointer   = const Type*;  //!< Pointer to a constant matrix value.

        
        StaticPanelMatrix()
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            v_.fill(Type {});
        }


        StaticPanelMatrix& operator=(Type val)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        template< typename MT    // Type of the right-hand side matrix
                , bool SO2 >      // Storage order of the right-hand side matrix
        StaticPanelMatrix& operator=(blaze::Matrix<MT, SO2> const& rhs);


        Type operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        Type& operator()(size_t i, size_t j)
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


        void pack(Type const * data, size_t lda)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(Type * data, size_t lda) const
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        Type * block(size_t i, size_t j)
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        Type const * block(size_t i, size_t j) const
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(AL) std::array<Type, panelRows_ * panelColumns_ * elementsPerPanel_> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return (panel_i * panelColumns_ + panel_j) * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };


    template <typename Type, size_t M, size_t N, size_t P, bool SO, size_t AL>
    template< typename MT    // Type of the right-hand side matrix
            , bool SO2 >      // Storage order of the right-hand side matrix
    inline StaticPanelMatrix<Type, M, N, P, SO, AL>& StaticPanelMatrix<Type, M, N, P, SO, AL>::operator=(blaze::Matrix<MT, SO2> const& rhs)
    {
        // using blaze::assign;

        // using TT = decltype( trans( *this ) );
        // using CT = decltype( ctrans( *this ) );
        // using IT = decltype( inv( *this ) );

        // if( (~rhs).rows() != M || (~rhs).columns() != N ) {
        //     BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static matrix" );
        // }

        // if( IsSame_v<MT,TT> && (~rhs).isAliased( this ) ) {
        //     transpose( typename IsSquare<This>::Type() );
        // }
        // else if( IsSame_v<MT,CT> && (~rhs).isAliased( this ) ) {
        //     ctranspose( typename IsSquare<This>::Type() );
        // }
        // else if( !IsSame_v<MT,IT> && (~rhs).canAlias( this ) ) {
        //     StaticPanelMatrix tmp( ~rhs );
        //     assign( *this, tmp );
        // }
        // else {
        //     if( IsSparseMatrix_v<MT> )
        //         reset();
        //     assign( *this, ~rhs );
        // }

        // BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

        assign(*this, ~rhs);

        return *this;
    }
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
            , bool SO
            , size_t AL >      // Alignment
    class Rand<StaticPanelMatrix<Type, M, N, P, SO, AL>>
    {
    public:
        //**Generate functions**************************************************************************
        /*!\name Generate functions */
        //@{
        inline const StaticPanelMatrix<Type, M, N, P, SO, AL> generate() const;

        template< typename Arg >
        inline const StaticPanelMatrix<Type, M, N, P, SO, AL> generate( const Arg& min, const Arg& max ) const;
        //@}
        //**********************************************************************************************

        //**Randomize functions*************************************************************************
        /*!\name Randomize functions */
        //@{
        inline void randomize( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix ) const;

        template< typename Arg >
        inline void randomize( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix, const Arg& min, const Arg& max ) const;
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    inline const StaticPanelMatrix<Type, M, N, P, SO, AL> Rand<StaticPanelMatrix<Type, M, N, P, SO, AL>>::generate() const
    {
        StaticPanelMatrix<Type, M, N, P, SO, AL> matrix;
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline const StaticPanelMatrix<Type, M, N, P, SO, AL>
    Rand<StaticPanelMatrix<Type, M, N, P, SO, AL>>::generate( const Arg& min, const Arg& max ) const
    {
        StaticPanelMatrix<Type, M, N, P, SO, AL> matrix;
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    inline void Rand< StaticPanelMatrix<Type, M, N, P, SO, AL> >::randomize( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix ) const
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    template< typename Arg >  // Min/max argument type
    inline void Rand< StaticPanelMatrix<Type, M, N, P, SO, AL> >::randomize( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix,
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    void makeSymmetric( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix )
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeSymmetric( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix, const Arg& min, const Arg& max )
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    void makeHermitian( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix )
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL      // Alignment
            , typename Arg >  // Min/max argument type
    void makeHermitian( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix, const Arg& min, const Arg& max )
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
            , size_t P  // Panel size
            , bool SO
            , size_t AL >      // Alignment
    void makePositiveDefinite( StaticPanelMatrix<Type, M, N, P, SO, AL>& matrix )
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