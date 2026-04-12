cmake_minimum_required(VERSION 3.20)
project(dna_gpu_features LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(dna_pipeline
    main.cpp
    prefix_scan.cu
    gc_content.cu
    homopolymer.cu
    motif_match.cu
    entropy.cu
    pipeline.cu
)

target_include_directories(dna_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(dna_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;80;86;90"
)

target_compile_options(dna_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(dna_pipeline PRIVATE CUDA::cudart)
