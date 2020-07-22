#pragma once


namespace blazefeo
{
    template< typename MT       // Type of the panel matrix
            , bool SO
            , size_t... CSAs >  // Compile time submatrix arguments
    class PanelSubmatrix
    {
    };
}