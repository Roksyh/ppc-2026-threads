#include "terekhov_d_seq_gauss_vert/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "terekhov_d_seq_gauss_vert/common/include/common.hpp"
#include "util/include/util.hpp"

namespace terekhov_d_seq_gauss_vert {

TerekhovDGaussVertOMP::TerekhovDGaussVertOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TerekhovDGaussVertOMP::ValidationImpl() {
  const auto &input = GetInput();
  if (input.width <= 0 || input.height <= 0) {
    return false;
  }
  if (static_cast<int>(input.data.size()) != input.width * input.height) {
    return false;
  }
  return true;
}

bool TerekhovDGaussVertOMP::PreProcessingImpl() {
  const auto &input = GetInput();
  width_ = input.width;
  height_ = input.height;

  GetOutput().width = width_;
  GetOutput().height = height_;
  GetOutput().data.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));

  int padded_width = width_ + 2;
  int padded_height = height_ + 2;
  padded_image_.resize(static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height));

  for (int row = 0; row < padded_height; ++row) {
    for (int col = 0; col < padded_width; ++col) {
      int src_x = col - 1;
      int src_y = row - 1;
      if (src_x < 0) { src_x = -src_x - 1; }
      if (src_x >= width_) { src_x = (2 * width_) - src_x - 1; }
      if (src_y < 0) { src_y = -src_y - 1; }
      if (src_y >= height_) { src_y = (2 * height_) - src_y - 1; }

      size_t padded_idx =
          (static_cast<size_t>(row) * static_cast<size_t>(padded_width)) + static_cast<size_t>(col);
      size_t src_idx =
          (static_cast<size_t>(src_y) * static_cast<size_t>(width_)) + static_cast<size_t>(src_x);
      padded_image_[padded_idx] = input.data[src_idx];
    }
  }
  return true;
}

void TerekhovDGaussVertOMP::ProcessPixel(OutType &output, int padded_width, int row, int col) {
  size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(width_)) + static_cast<size_t>(col);
  float sum = 0.0F;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int px = col + kx + 1;
      int py = row + ky + 1;
      int kernel_idx = ((ky + 1) * 3) + (kx + 1);
      size_t padded_idx =
          (static_cast<size_t>(py) * static_cast<size_t>(padded_width)) + static_cast<size_t>(px);
      sum += static_cast<float>(padded_image_[padded_idx]) *
             kGaussKernel[static_cast<size_t>(kernel_idx)];
    }
  }
  output.data[idx] = static_cast<int>(std::lround(sum));
}

void TerekhovDGaussVertOMP::ProcessBand(OutType &output, int padded_width, int band, int band_width) {
  int start_x = band * band_width;
  int end_x = (band == kNumBands - 1) ? width_ : ((band + 1) * band_width);
  for (int row = 0; row < height_; ++row) {
    for (int col = start_x; col < end_x; ++col) {
      ProcessPixel(output, padded_width, row, col);
    }
  }
}

void TerekhovDGaussVertOMP::ProcessBandsOMP(OutType &output) {
  int padded_width = width_ + 2;
  int band_width = std::max(width_ / kNumBands, 1);

#pragma omp parallel for default(none) shared(output, padded_width, band_width) \
    num_threads(ppc::util::GetNumThreads()) schedule(static)
  for (int band = 0; band < kNumBands; ++band) {
    ProcessBand(output, padded_width, band, band_width);
  }
}

bool TerekhovDGaussVertOMP::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  if (input.data.empty() || width_ <= 0 || height_ <= 0) {
    return false;
  }
  ProcessBandsOMP(output);
  return true;
}

bool TerekhovDGaussVertOMP::PostProcessingImpl() {
  return GetOutput().data.size() ==
         (static_cast<size_t>(GetOutput().width) * static_cast<size_t>(GetOutput().height));
}

}  // namespace terekhov_d_seq_gauss_vert
