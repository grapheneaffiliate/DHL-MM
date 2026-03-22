/**
 * Pybind11 C++ extension for sparse Lie bracket kernel.
 *
 * Core operation: result[k] += C[n] * x[I[n]] * y[J[n]]  for all n
 *
 * This is the sparse gather-multiply-scatter that replaces dense matrix multiply
 * for Lie algebra bracket computation using structure constants.
 *
 * Supports float32 and float64, single and batched, with backward pass
 * for PyTorch autograd integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ============================================================
// Single-vector sparse bracket (no OpenMP — overhead not worth it)
// ============================================================

template <typename T>
py::array_t<T> sparse_bracket_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> I,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> J,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> K,
    py::array_t<T, py::array::c_style | py::array::forcecast> C,
    int dim
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto I_buf = I.request();
    auto J_buf = J.request();
    auto K_buf = K.request();
    auto C_buf = C.request();

    if (x_buf.ndim != 1 || y_buf.ndim != 1)
        throw std::runtime_error("x and y must be 1-D arrays");
    if (I_buf.ndim != 1 || J_buf.ndim != 1 || K_buf.ndim != 1 || C_buf.ndim != 1)
        throw std::runtime_error("I, J, K, C must be 1-D arrays");

    ssize_t n_entries = I_buf.shape[0];
    if (J_buf.shape[0] != n_entries || K_buf.shape[0] != n_entries || C_buf.shape[0] != n_entries)
        throw std::runtime_error("I, J, K, C must have the same length");
    if (x_buf.shape[0] < dim || y_buf.shape[0] < dim)
        throw std::runtime_error("x and y must have length >= dim");

    const T* x_ptr = static_cast<const T*>(x_buf.ptr);
    const T* y_ptr = static_cast<const T*>(y_buf.ptr);
    const int32_t* I_ptr = static_cast<const int32_t*>(I_buf.ptr);
    const int32_t* J_ptr = static_cast<const int32_t*>(J_buf.ptr);
    const int32_t* K_ptr = static_cast<const int32_t*>(K_buf.ptr);
    const T* C_ptr = static_cast<const T*>(C_buf.ptr);

    auto result = py::array_t<T>(dim);
    auto res_buf = result.request();
    T* res_ptr = static_cast<T*>(res_buf.ptr);
    std::memset(res_ptr, 0, dim * sizeof(T));

    for (ssize_t n = 0; n < n_entries; n++) {
        res_ptr[K_ptr[n]] += C_ptr[n] * x_ptr[I_ptr[n]] * y_ptr[J_ptr[n]];
    }

    return result;
}

// ============================================================
// Batched sparse bracket — OpenMP over batch dimension
// ============================================================

template <typename T>
py::array_t<T> sparse_bracket_batched_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> I,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> J,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> K,
    py::array_t<T, py::array::c_style | py::array::forcecast> C,
    int dim
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto I_buf = I.request();
    auto J_buf = J.request();
    auto K_buf = K.request();
    auto C_buf = C.request();

    if (x_buf.ndim != 2 || y_buf.ndim != 2)
        throw std::runtime_error("x and y must be 2-D arrays (batch, dim)");
    if (I_buf.ndim != 1 || J_buf.ndim != 1 || K_buf.ndim != 1 || C_buf.ndim != 1)
        throw std::runtime_error("I, J, K, C must be 1-D arrays");

    ssize_t batch = x_buf.shape[0];
    ssize_t x_dim = x_buf.shape[1];
    ssize_t y_dim = y_buf.shape[1];
    ssize_t n_entries = I_buf.shape[0];

    if (y_buf.shape[0] != batch)
        throw std::runtime_error("x and y must have the same batch size");
    if (J_buf.shape[0] != n_entries || K_buf.shape[0] != n_entries || C_buf.shape[0] != n_entries)
        throw std::runtime_error("I, J, K, C must have the same length");
    if (x_dim < dim || y_dim < dim)
        throw std::runtime_error("x and y second dimension must be >= dim");

    const T* x_ptr = static_cast<const T*>(x_buf.ptr);
    const T* y_ptr = static_cast<const T*>(y_buf.ptr);
    const int32_t* I_ptr = static_cast<const int32_t*>(I_buf.ptr);
    const int32_t* J_ptr = static_cast<const int32_t*>(J_buf.ptr);
    const int32_t* K_ptr = static_cast<const int32_t*>(K_buf.ptr);
    const T* C_ptr = static_cast<const T*>(C_buf.ptr);

    auto result = py::array_t<T>({batch, static_cast<ssize_t>(dim)});
    auto res_buf = result.request();
    T* res_ptr = static_cast<T*>(res_buf.ptr);
    std::memset(res_ptr, 0, batch * dim * sizeof(T));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ssize_t b = 0; b < batch; b++) {
        const T* xb = x_ptr + b * x_dim;
        const T* yb = y_ptr + b * y_dim;
        T* rb = res_ptr + b * dim;

        for (ssize_t n = 0; n < n_entries; n++) {
            rb[K_ptr[n]] += C_ptr[n] * xb[I_ptr[n]] * yb[J_ptr[n]];
        }
    }

    return result;
}

// ============================================================
// Backward pass — single vector
// grad_x[i] = sum over entries where I[n]==i: C[n] * y[J[n]] * grad_out[K[n]]
// grad_y[j] = sum over entries where J[n]==j: C[n] * x[I[n]] * grad_out[K[n]]
// ============================================================

template <typename T>
std::pair<py::array_t<T>, py::array_t<T>> sparse_bracket_backward_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<T, py::array::c_style | py::array::forcecast> grad_output,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> I,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> J,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> K,
    py::array_t<T, py::array::c_style | py::array::forcecast> C,
    int dim
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto g_buf = grad_output.request();
    auto I_buf = I.request();
    auto J_buf = J.request();
    auto K_buf = K.request();
    auto C_buf = C.request();

    if (x_buf.ndim != 1 || y_buf.ndim != 1 || g_buf.ndim != 1)
        throw std::runtime_error("x, y, grad_output must be 1-D arrays");

    ssize_t n_entries = I_buf.shape[0];

    const T* x_ptr = static_cast<const T*>(x_buf.ptr);
    const T* y_ptr = static_cast<const T*>(y_buf.ptr);
    const T* g_ptr = static_cast<const T*>(g_buf.ptr);
    const int32_t* I_ptr = static_cast<const int32_t*>(I_buf.ptr);
    const int32_t* J_ptr = static_cast<const int32_t*>(J_buf.ptr);
    const int32_t* K_ptr = static_cast<const int32_t*>(K_buf.ptr);
    const T* C_ptr = static_cast<const T*>(C_buf.ptr);

    auto grad_x = py::array_t<T>(dim);
    auto grad_y = py::array_t<T>(dim);
    auto gx_buf = grad_x.request();
    auto gy_buf = grad_y.request();
    T* gx_ptr = static_cast<T*>(gx_buf.ptr);
    T* gy_ptr = static_cast<T*>(gy_buf.ptr);
    std::memset(gx_ptr, 0, dim * sizeof(T));
    std::memset(gy_ptr, 0, dim * sizeof(T));

    for (ssize_t n = 0; n < n_entries; n++) {
        T cg = C_ptr[n] * g_ptr[K_ptr[n]];
        gx_ptr[I_ptr[n]] += cg * y_ptr[J_ptr[n]];
        gy_ptr[J_ptr[n]] += cg * x_ptr[I_ptr[n]];
    }

    return std::make_pair(grad_x, grad_y);
}

// ============================================================
// Backward pass — batched
// ============================================================

template <typename T>
std::pair<py::array_t<T>, py::array_t<T>> sparse_bracket_backward_batched_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<T, py::array::c_style | py::array::forcecast> grad_output,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> I,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> J,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> K,
    py::array_t<T, py::array::c_style | py::array::forcecast> C,
    int dim
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto g_buf = grad_output.request();
    auto I_buf = I.request();
    auto J_buf = J.request();
    auto K_buf = K.request();
    auto C_buf = C.request();

    if (x_buf.ndim != 2 || y_buf.ndim != 2 || g_buf.ndim != 2)
        throw std::runtime_error("x, y, grad_output must be 2-D arrays (batch, dim)");

    ssize_t batch = x_buf.shape[0];
    ssize_t x_dim = x_buf.shape[1];
    ssize_t y_dim = y_buf.shape[1];
    ssize_t g_dim = g_buf.shape[1];
    ssize_t n_entries = I_buf.shape[0];

    const T* x_ptr = static_cast<const T*>(x_buf.ptr);
    const T* y_ptr = static_cast<const T*>(y_buf.ptr);
    const T* g_ptr = static_cast<const T*>(g_buf.ptr);
    const int32_t* I_ptr = static_cast<const int32_t*>(I_buf.ptr);
    const int32_t* J_ptr = static_cast<const int32_t*>(J_buf.ptr);
    const int32_t* K_ptr = static_cast<const int32_t*>(K_buf.ptr);
    const T* C_ptr = static_cast<const T*>(C_buf.ptr);

    auto grad_x = py::array_t<T>({batch, static_cast<ssize_t>(dim)});
    auto grad_y = py::array_t<T>({batch, static_cast<ssize_t>(dim)});
    auto gx_buf = grad_x.request();
    auto gy_buf = grad_y.request();
    T* gx_ptr = static_cast<T*>(gx_buf.ptr);
    T* gy_ptr = static_cast<T*>(gy_buf.ptr);
    std::memset(gx_ptr, 0, batch * dim * sizeof(T));
    std::memset(gy_ptr, 0, batch * dim * sizeof(T));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ssize_t b = 0; b < batch; b++) {
        const T* xb = x_ptr + b * x_dim;
        const T* yb = y_ptr + b * y_dim;
        const T* gb = g_ptr + b * g_dim;
        T* gxb = gx_ptr + b * dim;
        T* gyb = gy_ptr + b * dim;

        for (ssize_t n = 0; n < n_entries; n++) {
            T cg = C_ptr[n] * gb[K_ptr[n]];
            gxb[I_ptr[n]] += cg * yb[J_ptr[n]];
            gyb[J_ptr[n]] += cg * xb[I_ptr[n]];
        }
    }

    return std::make_pair(grad_x, grad_y);
}

// ============================================================
// Module definition — expose both float32 and float64 variants
// ============================================================

PYBIND11_MODULE(_csparse, m) {
    m.doc() = "C++ sparse Lie bracket kernel for DHL-MM";

    // float64 (default)
    m.def("sparse_bracket", &sparse_bracket_impl<double>,
          "Sparse Lie bracket (single vector, float64)",
          py::arg("x"), py::arg("y"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_batched", &sparse_bracket_batched_impl<double>,
          "Sparse Lie bracket (batched, float64)",
          py::arg("x"), py::arg("y"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_backward", &sparse_bracket_backward_impl<double>,
          "Sparse Lie bracket backward pass (single, float64)",
          py::arg("x"), py::arg("y"), py::arg("grad_output"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_backward_batched", &sparse_bracket_backward_batched_impl<double>,
          "Sparse Lie bracket backward pass (batched, float64)",
          py::arg("x"), py::arg("y"), py::arg("grad_output"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    // float32 variants
    m.def("sparse_bracket_f32", &sparse_bracket_impl<float>,
          "Sparse Lie bracket (single vector, float32)",
          py::arg("x"), py::arg("y"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_batched_f32", &sparse_bracket_batched_impl<float>,
          "Sparse Lie bracket (batched, float32)",
          py::arg("x"), py::arg("y"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_backward_f32", &sparse_bracket_backward_impl<float>,
          "Sparse Lie bracket backward pass (single, float32)",
          py::arg("x"), py::arg("y"), py::arg("grad_output"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));

    m.def("sparse_bracket_backward_batched_f32", &sparse_bracket_backward_batched_impl<float>,
          "Sparse Lie bracket backward pass (batched, float32)",
          py::arg("x"), py::arg("y"), py::arg("grad_output"),
          py::arg("I"), py::arg("J"), py::arg("K"), py::arg("C"),
          py::arg("dim"));
}
