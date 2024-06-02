#include "my_container.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cfloat>

#ifdef COMPILER_RT_HAS_FLOAT16
#define TYPE_FP16 _Float16
#else
#define TYPE_FP16 uint16_t
#endif

using fp16_t = uint16_t;

extern "C" void _mlir_ciface_forward(MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *);

// ----------------- Float16<->Float32 conversion utils ----------------------
extern "C" fp16_t __gnu_f2h_ieee(float f);
extern "C" float __gnu_h2f_ieee(fp16_t hf);

int main() {

  /// Initialize data containers
  fp16_t indata1[5] = {__gnu_f2h_ieee(1.f), __gnu_f2h_ieee(2.f), __gnu_f2h_ieee(7.f), __gnu_f2h_ieee(4.f), __gnu_f2h_ieee(5.f)};
  fp16_t indata2[5] = {__gnu_f2h_ieee(6.f), __gnu_f2h_ieee(7.f), __gnu_f2h_ieee(8.f), __gnu_f2h_ieee(9.f), __gnu_f2h_ieee(10.f)};
  long size[1] = {5};
  MemRef<fp16_t, 1> inputContainer1((fp16_t*)indata1, size, 0l);
  MemRef<fp16_t, 1> inputContainer2((fp16_t*)indata2, size, 0l);
  MemRef<fp16_t, 1> resultContainer({5}, __gnu_f2h_ieee(9.f));

  // check input
  fp16_t *input_data1 = (fp16_t*)inputContainer1.getData();
  std::cout << std::endl << (void*)(input_data1) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __gnu_h2f_ieee(input_data1[i]) << " ";
  }
  std::cout << std::endl;
  fp16_t *input_data2 = (fp16_t*)inputContainer2.getData();
  std::cout << std::endl << (void*)(input_data2) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __gnu_h2f_ieee(input_data2[i]) << " ";
  }
  std::cout << std::endl;

  // Execute the forward pass of the model.
  std::cout << "Start inference" << std::endl;
  _mlir_ciface_forward(&resultContainer, &inputContainer1, &inputContainer2);
  std::cout << "Finish inference" << std::endl;
  assert (resultContainer.sizes[0] == 5);
  assert (resultContainer.strides[0] == 1);

  // check output
  // assert(output_data != resultContainer.getData());
  fp16_t* output_data = (fp16_t*)resultContainer.getData();
  std::cout << std::endl << (void*)(output_data) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __gnu_h2f_ieee(output_data[i]) << " ";
  }
  std::cout << std::endl;

  return 0;
}