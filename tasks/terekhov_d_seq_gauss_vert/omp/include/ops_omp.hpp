#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "terekhov_d_seq_gauss_vert/common/include/common.hpp"

namespace terekhov_d_seq_gauss_vert {

class TerekhovDGaussVertOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit TerekhovDGaussVertOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ProcessBandsOMP(OutType &output);
  void ProcessBand(OutType &output, int padded_width, int band, int band_width,
                   const std::vector<int> &local_padded_image) const;
  void ProcessPixel(OutType &output, int padded_width, int row, int col,
                    const std::vector<int> &local_padded_image) const;

  int width_ = 0;
  int height_ = 0;
  static constexpr int kNumBands = 4;
  std::vector<int> padded_image_;
};

}  // namespace terekhov_d_seq_gauss_vert
