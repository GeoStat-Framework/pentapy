# cython: language_level=3
cdef double[::, ::1] c_penta_solver1(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int num_threads,
    int* info,
)

cdef double[::, ::1] c_penta_solver2(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int num_threads,
    int* info,
)
