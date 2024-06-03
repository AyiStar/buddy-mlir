#include <buddy/Core/Container.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

using fp16_t = uint16_t;

extern "C" void _mlir_ciface_forward(MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *);

// ----------------- Float16<->Float32 conversion builtins ----------------------
extern "C" fp16_t __gnu_f2h_ieee(float f);
extern "C" float __gnu_h2f_ieee(fp16_t hf);

static std::string container2string(const MemRef<fp16_t, 1>& container, int n) {
  std::ostringstream ss;
  for (int i = 0; i < n; i++) {
    ss << __gnu_h2f_ieee(container[i]);
    if (i < n - 1) {
      ss << " ";
    }
  }
  return ss.str();
}

// ---------------- 

int main() {

  constexpr int kSize = 5;

  /// Initialize data containers
  fp16_t indata1[kSize] = {0};
  fp16_t indata2[kSize] = {0};
  for (int i = 0; i < kSize; i++) {
    indata1[i] = __gnu_f2h_ieee(static_cast<float>(2 * i));
    indata2[i] = __gnu_f2h_ieee(static_cast<float>(2 * i + 1));
  }
  long sizes[1] = {kSize};
  MemRef<fp16_t, 1> inputContainer1((fp16_t*)indata1, sizes, 0l);
  MemRef<fp16_t, 1> inputContainer2((fp16_t*)indata2, sizes, 0l);
  MemRef<fp16_t, 1> resultContainer(sizes);

  // check input
  std::cout << "Input 1: " << container2string(inputContainer1, kSize) << std::endl;
  std::cout << "Input 2: " << container2string(inputContainer2, kSize) << std::endl;

  // Execute the forward pass of the model.
  std::cout << "Start inference" << std::endl;
  _mlir_ciface_forward(&resultContainer, &inputContainer1, &inputContainer2);
  std::cout << "Finish inference" << std::endl;

  // check output
  std::cout << "Output: " << container2string(resultContainer, kSize) << std::endl;

  return 0;
}