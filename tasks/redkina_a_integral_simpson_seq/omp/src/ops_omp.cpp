// redkina_a_integral_simpson_seq/omp/src/ops_omp.cpp
#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();
  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;
  result_ = 0.0;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  const size_t dim = a_.size();

  // Шаги сетки
  std::vector<double> h(dim);
  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
    h_prod *= h[i];
  }

  // Общее количество узлов (комбинаций индексов)
  // Используем знаковый тип для индекса цикла в OpenMP
  using index_t = long long;
  index_t total_nodes = 1;
  for (int ni : n_) {
    total_nodes *= static_cast<index_t>(ni + 1);
  }

  double global_sum = 0.0;

// Распараллеливание: каждый поток создаёт свои локальные векторы один раз
#pragma omp parallel
  {
    // Локальные для потока векторы (переиспользуются на всех итерациях потока)
    std::vector<double> local_point(dim);
    std::vector<int> local_indices(dim);

#pragma omp for reduction(+ : global_sum)
    for (index_t lin = 0; lin < total_nodes; ++lin) {
      // Преобразование линейного индекса в многомерные индексы
      index_t tmp = lin;
      double weight = 1.0;
      for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
        const int base = n_[d] + 1;
        const int idx = static_cast<int>(tmp % base);
        tmp /= base;
        local_indices[d] = idx;
        local_point[d] = a_[d] + static_cast<double>(idx) * h[d];

        // Коэффициент Симпсона для данного измерения
        int coeff;
        if (idx == 0 || idx == n_[d]) {
          coeff = 1;
        } else if (idx % 2 == 1) {
          coeff = 4;
        } else {
          coeff = 2;
        }
        weight *= static_cast<double>(coeff);
      }

      // Добавление вклада узла
      global_sum += weight * func_(local_point);
    }
  }

  // Итоговый результат: (h1*h2*...*hd / 3^d) * сумма
  const double denominator = std::pow(3.0, static_cast<double>(dim));
  result_ = (h_prod / denominator) * global_sum;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
