#pragma once

#include "melnik_i_radix_sort_int/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_radix_sort_int {

class MelnikIRadixSortIntOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit MelnikIRadixSortIntOMP(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSort(OutType& data);
  static int GetMaxValue(const OutType& data);
  static void ParallelCountingSort(OutType& data, int exp, int offset);
};

}  // namespace melnik_i_radix_sort_int
