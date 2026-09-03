#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*
Compile this file with

c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) SN.cpp SN_bindings.cpp -o SNcpp$(python3-config --extension-suffix)
*/

int mainupdate(int m, int n, int r, int maxiter, double maxtime, double *X, double *Wt, double *H, double *obj, double *time, int inneriter, double delta, int obj_compute);

int mainupdate_wrapper(
    int m, int n, int r, int maxiter, double maxtime,
    py::array_t<double> X,
    py::array_t<double> Wt,
    py::array_t<double> H,
    py::array_t<double> objlist,
    py::array_t<double> timelist,
    int inneriter, double delta, int obj_compute
) {
    return mainupdate(
        m, n, r, maxiter, maxtime,
        X.mutable_data(),
        Wt.mutable_data(),
        H.mutable_data(),
        objlist.mutable_data(),
        timelist.mutable_data(),
        inneriter, delta, obj_compute
    );
}

PYBIND11_MODULE(SNcpp, m) {
    m.def("run", &mainupdate_wrapper, "KL optimization");
}