#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/concatnew_layer.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {

template <typename Dtype>
__global__ void ConcatNew(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Dtype>
void ConcatNewLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs,cv_seg,cv_imgseg;
    this->data_transformer_->TransformInv(bottom[0], &cv_imgs);
    this->data_transformer_->TransformInv(bottom[1], &cv_seg);
    this->data_transformer_->TransformInv(bottom[3], &cv_imgseg);
    //vector<cv::Mat> new_imgs = AddChannels(cv_imgs, cv_seg);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, cv_seg, cv_imgseg, bottom[2], visualize_threshold_, colors,
        label_to_display_name_, save_file_);
#endif  // USE_OPENCV
  }
}

template <typename Dtype>
void ConcatNewLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatNewLayer);

}  // namespace caffe
