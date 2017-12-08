#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/cutwo_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void CutwoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CutwoParameter& param = this->layer_param_.cutwo_param();
  CHECK_EQ(bottom.size(), 1) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "cutwo axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of cutwo values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

template <typename Dtype>
void CutwoLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CutwoParameter& param = this->layer_param_.cutwo_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine cutwo offsets and the new shape post-cutwo.
  for (int i = 0; i < input_dim; ++i) {
    int cutwo_offset = 0;
    int new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      if (i==2) {
        new_size = bottom[0]->shape(i)/2;
      }
      else {
        new_size = bottom[0]->shape(i);
      }
      if (param.offset_size() == 1) {
        // If only one offset is given, all cutwos have the same offset.
        cutwo_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to cutwo, that is dimensions after the axis.
        cutwo_offset = param.offset(i - start_axis);
      }
      // Check that the cutwo and offset are within the dimension's bounds.
      //CHECK_GE(bottom[0]->shape(i) - cutwo_offset, bottom[1]->shape(i))
      //    << "the cutwo for dimension " << i << " is out-of-bounds with "
      //    << "size " << bottom[1]->shape(i) << " and offset " << cutwo_offset;
    }
    new_shape[i] = new_size;
    offsets[i] = cutwo_offset;
  }
  top[0]->Reshape(new_shape);
}

template <typename Dtype>
void CutwoLayer<Dtype>::cutwo_copy(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  if (cur_dim + 1 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursively
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      cutwo_copy(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last dimensions, which is stored continously in memory
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      // prepare index vector reduced(red) and with offsets(off)
      std::vector<int> ind_red(cur_dim, 0);
      std::vector<int> ind_off(cur_dim+1, 0);
      for (int j = 0; j < cur_dim; ++j) {
          ind_red[j] = indices[j];
          ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[cur_dim] = offsets[cur_dim];
      // do the copy
      if (is_forward) {
        caffe_copy(top[0]->shape(cur_dim),
            src_data + bottom[0]->offset(ind_off),
            dest_data + top[0]->offset(ind_red));
      } else {
        // in the backwards pass the src_data is top_diff
        // and the dest_data is bottom_diff
        caffe_copy(top[0]->shape(cur_dim),
            src_data + top[0]->offset(ind_red),
            dest_data + bottom[0]->offset(ind_off));
      }
    }
  }
}

template <typename Dtype>
void CutwoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  cutwo_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CutwoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CutwoLayer);
#endif

INSTANTIATE_CLASS(CutwoLayer);
REGISTER_LAYER_CLASS(Cutwo);

}  // namespace caffe
