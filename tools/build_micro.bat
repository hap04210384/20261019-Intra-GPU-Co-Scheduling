@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -gencode arch=compute_86,code=sm_86 -std=c++17 -Xcompiler /EHsc -o run\microbench.exe microbench_contention.cu
