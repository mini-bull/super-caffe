#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/concatnew_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatNewLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatNewParameter& concatnew_param = this->layer_param_.concatnew_param();
  CHECK(!(concatnew_param.has_axis() && concatnew_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
///////////////////////////////////////////////////////////////////////////
  const SaveOutputParameter& save_output_param =
      concatnew_param.save_output_param();
  output_directory_ = save_output_param.output_directory();
  need_save_ = output_directory_ == "" ? false : true;
  if (save_output_param.has_label_map_file()) {
    string label_map_file = save_output_param.label_map_file();
    if (label_map_file.empty()) {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
      need_save_ = false;
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  }
  visualize_ = concatnew_param.visualize();
  if (visualize_) {
    visualize_threshold_ = 0.6;
    if (concatnew_param.has_visualize_threshold()) {
      visualize_threshold_ = concatnew_param.visualize_threshold();
    }
    data_transformer_.reset(
        new DataTransformer<Dtype>(this->layer_param_.transform_param(),
                                   this->phase_));
    data_transformer_->InitRand();
    save_file_ = concatnew_param.save_file();
  }
}

template <typename Dtype>
void ConcatNewLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const ConcatNewParameter& concatnew_param = this->layer_param_.concatnew_param();
  if (concatnew_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concatnew_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concatnew_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size()-2; ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void ConcatNewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs,cv_seg,cv_imgseg;
    this->data_transformer_->TransformInv(bottom[0], &cv_imgs);
    this->data_transformer_->TransformInv(bottom[1], &cv_seg);
    this->data_transformer_->TransformInv(bottom[3], &cv_imgseg);

    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, cv_seg, cv_imgseg, bottom[2], visualize_threshold_, colors,
        label_to_display_name_, save_file_);
#endif  // USE_OPENCV
  }
}

template <typename Dtype>
void ConcatNewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ConcatNewLayer);
#endif

INSTANTIATE_CLASS(ConcatNewLayer);
REGISTER_LAYER_CLASS(ConcatNew);

}  // namespace caffe
