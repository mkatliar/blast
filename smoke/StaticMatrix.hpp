#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Panel.hpp>


namespace smoke
{
    template <typename T, size_t M, size_t N, size_t P = panelSize>
    class StaticMatrix
    {
    public:
        Panel<T, P> load(size_t i, size_t j) const
        {
            return Panel<T, P>(panelPtr(i, j));
        }


        void store(size_t i, size_t j, Panel<T, P> const& panel)
        {
            panel.store(panelPtr(i, j));
        }


        T operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        T& operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        size_t constexpr panelRows() const
        {
            return panelRows_;
        }


        size_t constexpr panelColumns() const
        {
            return panelColumns_;
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(Panel<T, P>::alignment) T v_[panelRows_ * panelColumns_ * elementsPerPanel_];


        T * panelPtr(size_t i, size_t j)
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        T const * panelPtr(size_t i, size_t j) const
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


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