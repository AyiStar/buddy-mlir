#include <buddy/Core/Container.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>


extern "C" void _mlir_ciface_forward(MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *);


int main() {
    /// Initialize data containers
  float indata1[5] = {1, 2, 3, 4, 5};
  float indata2[5] = {6, 7, 8, 9, 10};
  long size[1] = {5};
  MemRef<float, 1> inputContainer1((float*)indata1, size, 0l);
  MemRef<float, 1> inputContainer2((float*)indata2, size, 0l);
  MemRef<float, 1> resultContainer({5}, 9);

  // check input
  float *input_data1 = (float*)inputContainer1.getData();
  std::cout << std::endl << (void*)(input_data1) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << input_data1[i] << " ";
  }
  std::cout << std::endl;
  float *input_data2 = (float*)inputContainer2.getData();
  std::cout << std::endl << (void*)(input_data2) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << (input_data2[i]) << " ";
  }
  std::cout << std::endl;

  // check output
  float *output_data = (float*)resultContainer.getData();
  std::cout << std::endl << (void*)(output_data) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << (output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Execute the forward pass of the model.
  std::cout << "Start inference" << std::endl;
  _mlir_ciface_forward(&resultContainer, &inputContainer1, &inputContainer2);
  std::cout << "Finish inference" << std::endl;

  // check input again
  input_data1 = (float*)inputContainer1.getData();
  std::cout << std::endl << (void*)(input_data1) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << (input_data1[i]) << " ";
  }
  std::cout << std::endl;
  input_data2 = (float*)inputContainer2.getData();
  std::cout << std::endl << (void*)(input_data2) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << (input_data2[i]) << " ";
  }
  std::cout << std::endl;

  // check output
  // assert(output_data != resultContainer.getData());
  output_data = (float*)resultContainer.getData();
  std::cout << std::endl << (void*)(output_data) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << (output_data[i]) << " ";
  }
  std::cout << std::endl;

  return 0;
}