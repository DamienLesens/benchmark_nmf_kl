#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*
Compile this file with

c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) CCD.cpp CCD_bindings.cpp -o CCDcpp$(python3-config --extension-suffix)
*/

// Declare your existing function
int newKL(int n, int m, int k, int maxiter, double maxtime,
          double *V, double *W, double *H,
          int trace, double *objlist, double *timelist);

int newKL_wrapper(
    int n, int m, int k, int maxiter, double maxtime,
    py::array_t<double> V,
    py::array_t<double> W,
    py::array_t<double> H,
    int trace,
    py::array_t<double> objlist,
    py::array_t<double> timelist
) {
    return newKL(
        n, m, k, maxiter, maxtime,
        V.mutable_data(),
        W.mutable_data(),
        H.mutable_data(),
        trace,
        objlist.mutable_data(),
        timelist.mutable_data()
    );
}

PYBIND11_MODULE(CCDcpp, m) {
    m.def("run", &newKL_wrapper, "KL optimization");
}
