#include <buddy/Core/Container.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cfloat>

using fp16_t = uint16_t;

extern "C" void _mlir_ciface_forward(MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *, MemRef<fp16_t, 1> *);

// ----------------- Float16<->Float32 conversion utils ----------------------

// static inline float fp32_from_bits(uint32_t w) {
//     union {
//         uint32_t as_bits;
//         float as_value;
//     } fp32;
//     fp32.as_bits = w;
//     return fp32.as_value;
// }

// static inline uint32_t fp32_to_bits(float f) {
//     union {
//         float as_value;
//         uint32_t as_bits;
//     } fp32;
//     fp32.as_value = f;
//     return fp32.as_bits;
// }

// static fp16_t fp32_to_fp16(float f) {
// #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
//     const float scale_to_inf = 0x1.0p+112f;
//     const float scale_to_zero = 0x1.0p-110f;
// #else
//     const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
//     const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
// #endif
//     float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

//     const uint32_t w = fp32_to_bits(f);
//     const uint32_t shl1_w = w + w;
//     const uint32_t sign = w & UINT32_C(0x80000000);
//     uint32_t bias = shl1_w & UINT32_C(0xFF000000);
//     if (bias < UINT32_C(0x71000000)) {
//         bias = UINT32_C(0x71000000);
//     }

//     base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
//     const uint32_t bits = fp32_to_bits(base);
//     const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
//     const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
//     const uint32_t nonsign = exp_bits + mantissa_bits;
//     return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
// }

// static float fp16_to_fp32(fp16_t h) {
//     const uint32_t w = (uint32_t) h << 16;
//     const uint32_t sign = w & UINT32_C(0x80000000);
//     const uint32_t two_w = w + w;

//     const uint32_t exp_offset = UINT32_C(0xE0) << 23;
// #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
//     const float exp_scale = 0x1.0p-112f;
// #else
//     const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
// #endif
//     const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

//     const uint32_t magic_mask = UINT32_C(126) << 23;
//     const float magic_bias = 0.5f;
//     const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

//     const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
//     const uint32_t result = sign |
//         (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
//     return fp32_from_bits(result);
// }

// Union used to make the int/float aliasing explicit so we can access the raw
// bits.
union Float32Bits {
  uint32_t u;
  float f;
};

const uint32_t kF32MantiBits = 23;
const uint32_t kF32HalfMantiBitDiff = 13;
const uint32_t kF32HalfBitDiff = 16;
const Float32Bits kF32Magic = {113 << kF32MantiBits};
const uint32_t kF32HalfExpAdjust = (127 - 15) << kF32MantiBits;

// Constructs the 16 bit representation for a half precision value from a float
// value. This implementation is adapted from Eigen.
uint16_t float2half(float floatValue) {
  const Float32Bits inf = {255 << kF32MantiBits};
  const Float32Bits f16max = {(127 + 16) << kF32MantiBits};
  const Float32Bits denormMagic = {((127 - 15) + (kF32MantiBits - 10) + 1)
                                   << kF32MantiBits};
  uint32_t signMask = 0x80000000u;
  uint16_t halfValue = static_cast<uint16_t>(0x0u);
  Float32Bits f;
  f.f = floatValue;
  uint32_t sign = f.u & signMask;
  f.u ^= sign;

  if (f.u >= f16max.u) {
    const uint32_t halfQnan = 0x7e00;
    const uint32_t halfInf = 0x7c00;
    // Inf or NaN (all exponent bits set).
    halfValue = (f.u > inf.u) ? halfQnan : halfInf; // NaN->qNaN and Inf->Inf
  } else {
    // (De)normalized number or zero.
    if (f.u < kF32Magic.u) {
      // The resulting FP16 is subnormal or zero.
      //
      // Use a magic value to align our 10 mantissa bits at the bottom of the
      // float. As long as FP addition is round-to-nearest-even this works.
      f.f += denormMagic.f;

      halfValue = static_cast<uint16_t>(f.u - denormMagic.u);
    } else {
      uint32_t mantOdd =
          (f.u >> kF32HalfMantiBitDiff) & 1; // Resulting mantissa is odd.

      // Update exponent, rounding bias part 1. The following expressions are
      // equivalent to `f.u += ((unsigned int)(15 - 127) << kF32MantiBits) +
      // 0xfff`, but without arithmetic overflow.
      f.u += 0xc8000fffU;
      // Rounding bias part 2.
      f.u += mantOdd;
      halfValue = static_cast<uint16_t>(f.u >> kF32HalfMantiBitDiff);
    }
  }

  halfValue |= static_cast<uint16_t>(sign >> kF32HalfBitDiff);
  return halfValue;
}

// Converts the 16 bit representation of a half precision value to a float
// value. This implementation is adapted from Eigen.
float half2float(uint16_t halfValue) {
  const uint32_t shiftedExp =
      0x7c00 << kF32HalfMantiBitDiff; // Exponent mask after shift.

  // Initialize the float representation with the exponent/mantissa bits.
  Float32Bits f = {
      static_cast<uint32_t>((halfValue & 0x7fff) << kF32HalfMantiBitDiff)};
  const uint32_t exp = shiftedExp & f.u;
  f.u += kF32HalfExpAdjust; // Adjust the exponent

  // Handle exponent special cases.
  if (exp == shiftedExp) {
    // Inf/NaN
    f.u += kF32HalfExpAdjust;
  } else if (exp == 0) {
    // Zero/Denormal?
    f.u += 1 << kF32MantiBits;
    f.f -= kF32Magic.f;
  }

  f.u |= (halfValue & 0x8000) << kF32HalfBitDiff; // Sign bit.
  return f.f;
}

// utils for debug
static std::string half2string(fp16_t hf) {
  std::stringstream stream;
  stream << std::hex << hf;
  std::string result( stream.str() );
  return result;
}


// ----------------- Workaround for undefined symbols ------------------- //

#define ATTR_WEAK __attribute__((__weak__))
// Provide a float->float16 conversion routine in case the runtime doesn't have
// one.
extern "C" fp16_t ATTR_WEAK __truncsfhf2(float f) {
  fp16_t hf = float2half(f);
  // The output can be a float type, bitcast it from uint16_t.
  fp16_t ret = 0;
  std::memcpy(&ret, &hf, sizeof(hf));
  std::cout << "[32to16: " << f << "->" << half2string(ret) << "]";
  return ret;
}

// Provide a float16->float conversion routine in case the runtime doesn't have
// one.
extern "C" float ATTR_WEAK __extendhfsf2(fp16_t hf) {
  float f = half2float(hf);
  std::cout << "[16to32: " << half2string(hf) << "->" << f << "]";
  return f;
}

int main() {

  /// Initialize data containers
  fp16_t indata1[5] = {__truncsfhf2(1.f), __truncsfhf2(2.f), __truncsfhf2(3.f), __truncsfhf2(4.f), __truncsfhf2(5.f)};
  fp16_t indata2[5] = {__truncsfhf2(6.f), __truncsfhf2(7.f), __truncsfhf2(8.f), __truncsfhf2(9.f), __truncsfhf2(10.f)};
  long size[1] = {5};
  MemRef<fp16_t, 1> inputContainer1((fp16_t*)indata1, size, 0l);
  MemRef<fp16_t, 1> inputContainer2((fp16_t*)indata2, size, 0l);
  MemRef<fp16_t, 1> resultContainer({5}, __truncsfhf2(9.f));

  // check input
  fp16_t *input_data1 = (fp16_t*)inputContainer1.getData();
  std::cout << std::endl << (void*)(input_data1) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(input_data1[i]) << " ";
  }
  std::cout << std::endl;
  fp16_t *input_data2 = (fp16_t*)inputContainer2.getData();
  std::cout << std::endl << (void*)(input_data2) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(input_data2[i]) << " ";
  }
  std::cout << std::endl;

  // check output
  fp16_t *output_data = (fp16_t*)resultContainer.getData();
  std::cout << std::endl << (void*)(output_data) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Execute the forward pass of the model.
  std::cout << "Start inference" << std::endl;
  _mlir_ciface_forward(&resultContainer, &inputContainer1, &inputContainer2);
  std::cout << "Finish inference" << std::endl;

  // check input again
  input_data1 = (fp16_t*)inputContainer1.getData();
  std::cout << std::endl << (void*)(input_data1) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(input_data1[i]) << " ";
  }
  std::cout << std::endl;
  input_data2 = (fp16_t*)inputContainer2.getData();
  std::cout << std::endl << (void*)(input_data2) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(input_data2[i]) << " ";
  }
  std::cout << std::endl;

  // check output
  // assert(output_data != resultContainer.getData());
  output_data = (fp16_t*)resultContainer.getData();
  std::cout << std::endl << (void*)(output_data) << ": ";
  for (int i = 0; i < 5; i++) {
    std::cout << __extendhfsf2(output_data[i]) << " ";
  }
  std::cout << std::endl;

  return 0;
}