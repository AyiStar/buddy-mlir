add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/f16pow.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/import-f16pow.py
  COMMENT "Generating f16pow.mlir..."
)

add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/f16pow.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/f16pow.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          tee ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/mlir-opt-ret.mlir |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-paralell-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        tee ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/buddy-opt-ret.mlir |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        tee ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/mlir-trans-ret.llvm |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as |
        ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${BUDDY_EXAMPLES_DIR}/../examples/BuddyF16pow/f16pow.o
  DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/f16pow.mlir
  COMMENT "Building f16pow.o "
  VERBATIM)

add_library(F16POW STATIC ${BUDDY_EXAMPLES_DIR}/BuddyF16pow/f16pow.o)
target_compile_options(F16POW PUBLIC -g)

# SET_SOURCE_FILES_PROPERTIES(
#   template.o
#   PROPERTIES
#   EXTERNAL_OBJECT true
#   GENERATED true)

SET_TARGET_PROPERTIES(
  F16POW
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-f16pow-run f16pow-main.cpp)
target_link_directories(buddy-f16pow-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_compile_options(buddy-f16pow-run PUBLIC -g)

set(BUDDY_F16POW_LIBS
  F16POW
  mlir_c_runner_utils
  omp
)

target_link_libraries(buddy-f16pow-run ${BUDDY_F16POW_LIBS})
