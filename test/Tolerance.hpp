#pragma once


namespace blazefeo :: testing
{
    template <typename ET>
    ET absTol();


    template <typename ET>
    ET relTol();


    template <>
    inline double absTol<double>()
    {
        return 1e-11;
    }


    template <>
    inline double relTol<double>()
    {
        return 1e-11;
    }


    template <>
    inline float absTol<float>()
    {
        return 1e-5f;
    }


    template <>
    inline float relTol<float>()
    {
        return 1e-4f;
    }
}