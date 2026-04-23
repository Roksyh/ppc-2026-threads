#include "terekhov_d_gauss_vert/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "terekhov_d_gauss_vert/common/include/common.hpp"

namespace terekhov_d_gauss_vert {

namespace {

inline void ProcessPixel(OutType &output, const std::vector<int> &padded_image, int padded_width, 
                         int width, int row, int col) {
  size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
  float sum = 0.0F;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int px = col + kx + 1;
      int py = row + ky + 1;
      int kernel_idx = ((ky + 1) * 3) + (kx + 1);
      size_t padded_idx = (static_cast<size_t>(py) * static_cast<size_t>(padded_width)) + static_cast<size_t>(px);
      sum += static_cast<float>(padded_image[padded_idx]) * kGaussKernel[static_cast<size_t>(kernel_idx)];
    }
  }
  output.data[idx] = static_cast<int>(std::lround(sum));
}

void ProcessBandOMP(OutType &output, const std::vector<int> &padded_image, int padded_width, 
                    int width, int /*height*/, int start_row, int end_row) {
  const int num_bands = 4;
  const int band_width = std::max(width / num_bands, 1);
  
#pragma omp parallel for default(none) shared(output, padded_image, padded_width, width, start_row, end_row, band_width, num_bands) schedule(static)
  for (int band = 0; band < num_bands; ++band) {
    int start_x = band * band_width;
    int end_x = (band == num_bands - 1) ? width : ((band + 1) * band_width);
    for (int row = start_row; row < end_row; ++row) {
      for (int col = start_x; col < end_x; ++col) {
        ProcessPixel(output, padded_image, padded_width, width, row, col);
      }
    }
  }
}

OutType SolveALL(const std::vector<int> &padded_image, int width, int height, 
                 int start_row, int end_row) {
  OutType output;
  output.width = width;
  output.height = height;
  output.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
  
  const int padded_width = width + 2;
  
  ProcessBandOMP(output, padded_image, padded_width, width, height, start_row, end_row);
  
  return output;
}

void BroadcastImageData(int rank, int &width, int &height, int &total_pixels, 
                        std::vector<int> &all_data, const InType &input) {
  if (rank == 0) {
    width = input.width;
    height = input.height;
    total_pixels = width * height;
    all_data = input.data;
  }
  
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_pixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank != 0) {
    all_data.resize(static_cast<size_t>(total_pixels));
  }
  
  MPI_Bcast(all_data.data(), total_pixels, MPI_INT, 0, MPI_COMM_WORLD);
}

int MirrorCoordinate(int coord, int max_val) {
  if (coord < 0) {
    return -coord - 1;
  }
  if (coord >= max_val) {
    return (2 * max_val) - coord - 1;
  }
  return coord;
}

void FillPaddedPixel(std::vector<int> &padded_image, const std::vector<int> &local_data,
                     int width, int height, int local_start, int local_height,
                     int row, int col, int padded_width) {
  int src_x = MirrorCoordinate(col - 1, width);
  int src_y = MirrorCoordinate((local_start + row) - 1, height);
  
  int local_y = src_y - local_start;
  size_t padded_idx = (static_cast<size_t>(row) * static_cast<size_t>(padded_width)) + static_cast<size_t>(col);
  
  if (local_y >= 0 && local_y < local_height) {
    size_t local_idx = (static_cast<size_t>(local_y) * static_cast<size_t>(width)) + static_cast<size_t>(src_x);
    padded_image[padded_idx] = local_data[local_idx];
  } else {
    padded_image[padded_idx] = 0;
  }
}

void CreatePaddedImage(const std::vector<int> &local_data, int width, int height, 
                       int local_start, int local_height, std::vector<int> &padded_image) {
  int padded_width = width + 2;
  int padded_height = local_height + 2;
  padded_image.resize(static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height));
  
  for (int row = 0; row < padded_height; ++row) {
    for (int col = 0; col < padded_width; ++col) {
      FillPaddedPixel(padded_image, local_data, width, height, local_start, local_height, 
                      row, col, padded_width);
    }
  }
}

void CopyLocalResultToOutput(const OutType &local_output, int width, int start_row, int end_row, 
                             std::vector<int> &output_data) {
  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t out_idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      output_data[out_idx] = local_output.data[out_idx];
    }
  }
}

void ReceiveProcData(int width, int proc_start, int proc_end, std::vector<int> &output_data) {
  int proc_rows = proc_end - proc_start;
  std::vector<int> proc_data(static_cast<size_t>(proc_rows) * static_cast<size_t>(width));
  MPI_Recv(proc_data.data(), proc_rows * width, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  for (int row = 0; row < proc_rows; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t out_idx = (static_cast<size_t>(proc_start + row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      output_data[out_idx] = proc_data[(static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col)];
    }
  }
}

void SendLocalData(const OutType &local_output, int width, int start_row, int end_row) {
  int local_rows = end_row - start_row;
  std::vector<int> send_data(static_cast<size_t>(local_rows) * static_cast<size_t>(width));
  
  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t out_idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      send_data[(static_cast<size_t>(row - start_row) * static_cast<size_t>(width)) + static_cast<size_t>(col)] = 
          local_output.data[out_idx];
    }
  }
  
  MPI_Send(send_data.data(), local_rows * width, MPI_INT, 0, 1, MPI_COMM_WORLD);
}

void GatherResultsToRoot(int rank, int size, const OutType &local_output, 
                         int width, int /*height*/, int start_row, int end_row,
                         int rows_per_proc, int remainder, OutType &output) {
  if (rank == 0) {
    output.width = width;
    output.height = static_cast<int>(output.data.size()) / width;
    
    CopyLocalResultToOutput(local_output, width, start_row, end_row, output.data);
    
    for (int proc = 1; proc < size; ++proc) {
      int proc_start = (proc * rows_per_proc) + std::min(proc, remainder);
      int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
      ReceiveProcData(width, proc_start, proc_end, output.data);
    }
  } else {
    SendLocalData(local_output, width, start_row, end_row);
  }
}

}  // namespace

TerekhovDGaussVertALL::TerekhovDGaussVertALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TerekhovDGaussVertALL::ValidationImpl() {
  const auto &input = GetInput();
  
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  bool valid = true;
  if (rank == 0) {
    if (input.width <= 0 || input.height <= 0) {
      valid = false;
    }
    if (static_cast<int>(input.data.size()) != input.width * input.height) {
      valid = false;
    }
  }
  
  MPI_Bcast(&valid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
  return valid;
}

bool TerekhovDGaussVertALL::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (rank == 0) {
    const auto &input = GetInput();
    width_ = input.width;
    height_ = input.height;
  }
  
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  return true;
}

bool TerekhovDGaussVertALL::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  std::vector<int> all_data;
  int total_pixels = 0;
  BroadcastImageData(rank, width_, height_, total_pixels, all_data, GetInput());
  
  int rows_per_proc = height_ / size;
  int remainder = height_ % size;
  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  
  int local_start = std::max(0, start_row - 1);
  int local_end = std::min(height_, end_row + 1);
  int local_height = local_end - local_start;
  
  std::vector<int> local_data(all_data.begin() + static_cast<std::ptrdiff_t>(local_start) * static_cast<std::ptrdiff_t>(width_),
                               all_data.begin() + static_cast<std::ptrdiff_t>(local_end) * static_cast<std::ptrdiff_t>(width_));
  
  CreatePaddedImage(local_data, width_, height_, local_start, local_height, padded_image_);
  
  OutType local_output = SolveALL(padded_image_, width_, height_, start_row, end_row);
  
  auto &output = GetOutput();
  output.data.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));
  GatherResultsToRoot(rank, size, local_output, width_, height_, start_row, end_row, 
                      rows_per_proc, remainder, output);
  
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool TerekhovDGaussVertALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  bool result = true;
  if (rank == 0) {
    result = GetOutput().data.size() == (static_cast<size_t>(GetOutput().width) * static_cast<size_t>(GetOutput().height));
  }
  
  MPI_Bcast(&result, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
  return result;
}
//1111
}  // namespace terekhov_d_gauss_vert
