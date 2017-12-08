#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/cut_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void CutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CutParameter& param = this->layer_param_.cut_param();
  CHECK_EQ(bottom.size(), 1) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "cut axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of cut values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

template <typename Dtype>
void CutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CutParameter& param = this->layer_param_.cut_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  offsets_cut = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine cut offsets and the new shape post-cut.
  for (int i = 0; i < input_dim; ++i) {
    int cut_offset = 0;
    int offset_cut = 0;
    int new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      if (i==2) {
        new_size = bottom[0]->shape(i)/2;
        offset_cut = 512;
      }
      else {
        new_size = bottom[0]->shape(i);
        offset_cut = 0;
      }
      if (param.offset_size() == 1) {
        // If only one offset is given, all cuts have the same offset.
        cut_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to cut, that is dimensions after the axis.
        cut_offset = param.offset(i - start_axis);
      }
      // Check that the cut and offset are within the dimension's bounds.
      //CHECK_GE(bottom[0]->shape(i) - cut_offset, bottom[1]->shape(i))
      //    << "the cut for dimension " << i << " is out-of-bounds with "
      //    << "size " << bottom[1]->shape(i) << " and offset " << cut_offset;
    }
    new_shape[i] = new_size;
    offsets[i] = cut_offset;
    offsets_cut[i] = offset_cut;
  }
  top[0]->Reshape(new_shape);
  top[1]->Reshape(new_shape);
}

template <typename Dtype>
void CutLayer<Dtype>::cut_copy(const vector<Blob<Dtype>*>& bottom,
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
      cut_copy(bottom, top, offsets, indices, cur_dim+1,
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
void CutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top[1]->mutable_cpu_data();
  cut_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
  cut_copy(bottom, top, offsets_cut, indices, 0, bottom_data, top_data_1, true);
}

template <typename Dtype>
void CutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CutLayer);
#endif

INSTANTIATE_CLASS(CutLayer);
REGISTER_LAYER_CLASS(Cut);

}  // namespace caffe
