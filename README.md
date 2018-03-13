# matrix
C++ 11 simple matrix class

This very simple header-only matrix class is designed for efficient "matrix view" operations, sometimes needed for certain algorithms. 
Data are stored in row major order, and can easily be used in BLAS or Intel MKL routines. Few routines using BLAS are defined in `matrix_lpack_blas.h`. 
The file ` matrix_lpack .h` contains some hard-coded routines, which are deprecated.  

The solution is in Microsoft Visual Studio 2017 (yes CMAKE would be nice). 
There are few thing to setup to use full advantege of numerical computation:  
 - download and setup Intel® Math Kernel Library (or OpenBLAS) and set the `BlasDir` in `Property Manager` ->`User macros`
 - copy files `mkl_core.dll, mkl_def.dll, mkl_sequential.dll` from for example `d:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\redist\intel64_win\mkl\` to your executable
 - to use unit tests edit path to `gtest` under `Configuration Properties -> Linker -> Input -> Additional library directories`
