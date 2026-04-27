cmake_minimum_required(VERSION 3.20)
project(dna_gpu_features LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(PROJECT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DNA_PIPELINE_SOURCES
    ${PROJECT_SRC_DIR}/main.cpp
    ${PROJECT_SRC_DIR}/prefix_scan.cu
    ${PROJECT_SRC_DIR}/gc_content.cu
    ${PROJECT_SRC_DIR}/homopolymer.cu
    ${PROJECT_SRC_DIR}/motif_match.cu
    ${PROJECT_SRC_DIR}/entropy.cu
    ${PROJECT_SRC_DIR}/pipeline.cu
)

add_executable(dna_pipeline
    ${DNA_PIPELINE_SOURCES}
)

add_executable(dna_pipeline_parallel
    ${DNA_PIPELINE_SOURCES}
)

add_executable(dna_pipeline_sequential
    ${DNA_PIPELINE_SOURCES}
)

target_include_directories(dna_pipeline PRIVATE ${PROJECT_SRC_DIR})
target_include_directories(dna_pipeline_parallel PRIVATE ${PROJECT_SRC_DIR})
target_include_directories(dna_pipeline_sequential PRIVATE ${PROJECT_SRC_DIR})

set_target_properties(dna_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;80;86;90"
)

set_target_properties(dna_pipeline_parallel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;80;86;90"
)

set_target_properties(dna_pipeline_sequential PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;80;86;90"
)

target_compile_options(dna_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(dna_pipeline_parallel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(dna_pipeline_sequential PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(dna_pipeline PRIVATE CUDA::cudart)
target_link_libraries(dna_pipeline_parallel PRIVATE CUDA::cudart)
target_link_libraries(dna_pipeline_sequential PRIVATE CUDA::cudart)

target_compile_definitions(dna_pipeline_parallel PRIVATE DNA_FORCE_PARALLEL_MODE)
target_compile_definitions(dna_pipeline_sequential PRIVATE DNA_FORCE_SEQUENTIAL_MODE)
