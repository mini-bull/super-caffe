#include <algorithm>
#include <vector>

#include "caffe/layers/sliceseg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SlicesegLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SlicesegParameter& sliceseg_param = this->layer_param_.sliceseg_param();
  CHECK(!(sliceseg_param.has_axis() && sliceseg_param.has_sliceseg_dim()))
      << "Either axis or sliceseg_dim should be specified; not both.";
  sliceseg_point_.clear();
  std::copy(sliceseg_param.sliceseg_point().begin(),
      sliceseg_param.sliceseg_point().end(),
      std::back_inserter(sliceseg_point_));
}

template <typename Dtype>
void SlicesegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const SlicesegParameter& sliceseg_param = this->layer_param_.sliceseg_param();
  if (sliceseg_param.has_sliceseg_dim()) {
    sliceseg_axis_ = static_cast<int>(sliceseg_param.sliceseg_dim());
    // Don't allow negative indexing for sliceseg_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(sliceseg_axis_, 0) << "casting sliceseg_dim from uint32 to int32 "
        << "produced negative result; sliceseg_dim must satisfy "
        << "0 <= sliceseg_dim < " << kMaxBlobAxes;
    CHECK_LT(sliceseg_axis_, num_axes) << "sliceseg_dim out of range.";
  } else {
    sliceseg_axis_ = bottom[0]->CanonicalAxisIndex(sliceseg_param.axis());
  }
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_sliceseg_axis = bottom[0]->shape(sliceseg_axis_);
  num_slicesegs_ = bottom[0]->count(0, sliceseg_axis_);
  sliceseg_size_ = bottom[0]->count(sliceseg_axis_ + 1);
  int count = 0;
  if (sliceseg_point_.size() != 0) {
    //CHECK_EQ(sliceseg_point_.size(), top.size() - 1);
    //CHECK_LE(top.size(), bottom_sliceseg_axis);
    int prev = 0;
    vector<int> slicesegs;
    for (int i = 0; i < sliceseg_point_.size(); ++i) {
      CHECK_GT(sliceseg_point_[i], prev);
      slicesegs.push_back(sliceseg_point_[i] - prev);
      prev = sliceseg_point_[i];
    }
    slicesegs.push_back(bottom_sliceseg_axis - prev);
    for (int i = 0; i < 1; ++i) {
      top_shape[sliceseg_axis_] = slicesegs[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  } else {
    //CHECK_EQ(bottom_sliceseg_axis % top.size(), 0)
    //    << "Number of top blobs (" << top.size() << ") should evenly "
    //    << "divide input sliceseg axis (" << bottom_sliceseg_axis << ")";
    top_shape[sliceseg_axis_] = bottom_sliceseg_axis / 2;
    for (int i = 0; i < 1; ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  }
  //CHECK_EQ(count, bottom[0]->count());
  //if (top.size() == 1) {
  //  top[0]->ShareData(*bottom[0]);
  //}
}

template <typename Dtype>
void SlicesegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //if (top.size() == 1) { return; }
  int offset_sliceseg_axis = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_sliceseg_axis = bottom[0]->shape(sliceseg_axis_);
  for (int i = 0; i < 1; ++i) {
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int top_sliceseg_axis = top[i]->shape(sliceseg_axis_);
    for (int n = 0; n < num_slicesegs_; ++n) {
      const int top_offset = n * top_sliceseg_axis * sliceseg_size_;
      const int bottom_offset =
          (n * bottom_sliceseg_axis + offset_sliceseg_axis) * sliceseg_size_;
      caffe_copy(top_sliceseg_axis * sliceseg_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_sliceseg_axis += top_sliceseg_axis;
  }
}

template <typename Dtype>
void SlicesegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SlicesegLayer);
#endif

INSTANTIATE_CLASS(SlicesegLayer);
REGISTER_LAYER_CLASS(Sliceseg);

}  // namespace caffe
