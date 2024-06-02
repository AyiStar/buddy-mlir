add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/import-f16add.py
  COMMENT "Generating f16add.mlir..."
)

set(BUDDY_OPT_OPTIONS -one-shot-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -lower-affine -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize   -arith-expand -convert-arith-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -llvm-request-c-wrappers -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts
)

add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/mlir-opt-ret-f16add.mlir |
          ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_OPT_OPTIONS} |
        tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/buddy-opt-ret-f16add.mlir |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.llvm |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as |
        ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0 -mattr=-sse2
          -o ${BUDDY_EXAMPLES_DIR}/../examples/BuddyF16Test/f16add.o
  DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.mlir
  COMMENT "Building f16add.o "
  VERBATIM
)

add_library(F16ADD STATIC ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16add.o)

SET_TARGET_PROPERTIES(
  F16ADD
  PROPERTIES
  LINKER_LANGUAGE C
)

add_executable(buddy-f16add-run f16add-main.cpp fp16_h2f.cpp fp16_f2h.cpp)
target_link_directories(buddy-f16add-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_include_directories(buddy-f16add-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR}/../../../)

set(BUDDY_F16ADD_LIBS
  F16ADD
  mlir_c_runner_utils
)

target_link_libraries(buddy-f16add-run ${BUDDY_F16ADD_LIBS})