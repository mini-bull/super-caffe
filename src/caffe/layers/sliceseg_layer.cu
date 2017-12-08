#include <vector>

#include "caffe/layers/sliceseg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Sliceseg(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slicesegs, const int sliceseg_size,
    const int bottom_sliceseg_axis, const int top_sliceseg_axis,
    const int offset_sliceseg_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_sliceseg_size = sliceseg_size * top_sliceseg_axis;
    const int sliceseg_num = index / total_sliceseg_size;
    const int sliceseg_index = index % total_sliceseg_size;
    const int bottom_index = sliceseg_index +
        (sliceseg_num * bottom_sliceseg_axis + offset_sliceseg_axis) * sliceseg_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

template <typename Dtype>
void SlicesegLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //if (top.size() == 1) { return; }
  int offset_sliceseg_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_sliceseg_axis = bottom[0]->shape(sliceseg_axis_);
  const bool kForward = true;
  for (int i = 0; i < 1; ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_sliceseg_axis = top[i]->shape(sliceseg_axis_);
    const int top_sliceseg_size = top_sliceseg_axis * sliceseg_size_;
    const int nthreads = top_sliceseg_size * num_slicesegs_;
    Sliceseg<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slicesegs_, sliceseg_size_,
        bottom_sliceseg_axis, top_sliceseg_axis, offset_sliceseg_axis, top_data);
    offset_sliceseg_axis += top_sliceseg_axis;
  }
}

template <typename Dtype>
void SlicesegLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(SlicesegLayer);

}  // namespace caffe
