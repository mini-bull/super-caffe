#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/gridgenerator_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void GridGeneratorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 // const GridGeneratorParameter& param = this->layer_param_.GridGenerator_param();
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GridGeneratorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const vector<int> bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int bottom_count = bottom[0]->count(); 

  const Dtype alpha = -1.0;//workspace_

  Blob<Dtype> grid_term_a_;
  Blob<Dtype> grid_term_b_;
  Blob<Dtype> term_a_;
  Blob<Dtype> term_b_;
  Blob<Dtype> grid_dst_a;
  Blob<Dtype> grid_dst_b;

  grid_term_a_.Reshape(bottom_shape[0],1,1,bottom_shape[3]);
  grid_term_b_.Reshape(bottom_shape[0],1,1,bottom_shape[2]);
  term_a_.Reshape(bottom_shape[0],1,1,bottom_shape[2]);
  term_b_.Reshape(bottom_shape[0],1,1,bottom_shape[3]);
  grid_dst_a.Reshape(bottom_shape[0],1,bottom_shape[2], bottom_shape[3]);
  grid_dst_b.Reshape(bottom_shape[0],1,bottom_shape[2], bottom_shape[3]);


  Dtype* grid_term_a_ptr = grid_term_a_.mutable_cpu_data();
  Dtype* grid_term_b_ptr = grid_term_b_.mutable_cpu_data();
  Dtype* term_a_ptr = term_a_.mutable_cpu_data();
  Dtype* term_b_ptr = term_b_.mutable_cpu_data();
  for (int o = 0; o<bottom_shape[3]; o++){
    grid_term_a_ptr[o] = o%bottom_shape[3];
    term_b_ptr[o] = 1;
  }

  for (int n = 0; n<bottom_shape[2]; n++){
    grid_term_b_ptr[n] = n%bottom_shape[2];
    term_a_ptr[n] = 1;
  }

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom_shape[2], bottom_shape[3], 1, (Dtype)1.,  
        term_a_.cpu_data(), grid_term_a_.cpu_data(), (Dtype)1., grid_dst_a.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom_shape[2], bottom_shape[3], 1, (Dtype)1.,  
        grid_term_b_.cpu_data(), term_b_.cpu_data(), (Dtype)1., grid_dst_b.mutable_cpu_data());

  Dtype workspace_term_a =  (bottom_shape[3]-1.0) / 2.0;
  Dtype workspace_term_b =  (bottom_shape[2]-1.0) / 2.0;

  Blob<Dtype> workspace_a;
  Blob<Dtype> workspace_b;
  workspace_a.Reshape(bottom_shape[0],1,bottom_shape[2], bottom_shape[3]);
  workspace_b.Reshape(bottom_shape[0],1,bottom_shape[2], bottom_shape[3]);
  Dtype *workspace_a_ptr = workspace_a.mutable_cpu_data();
  Dtype *workspace_b_ptr = workspace_b.mutable_cpu_data();

  for (int i = 0; i<workspace_a.count();i++) {
    workspace_a_ptr[i] = workspace_term_a;
    workspace_b_ptr[i] = workspace_term_b;
  }

  grid_dst_.Reshape(bottom_shape[0],2,bottom_shape[2], bottom_shape[3]);
  workspace_.Reshape(bottom_shape[0],2,bottom_shape[2], bottom_shape[3]);
  Dtype *grid_dst_ptr = grid_dst_.mutable_cpu_data();
  Dtype *workspace_ptr = workspace_.mutable_cpu_data();
  Dtype *grid_dst_a_ptr = grid_dst_a.mutable_cpu_data();
  Dtype *grid_dst_b_ptr = grid_dst_b.mutable_cpu_data();

  for (int j = 0; j<bottom_shape[0]*bottom_shape[2]*bottom_shape[3];j++){
    grid_dst_ptr[j] = grid_dst_a_ptr[j];
  }
  for (int k = bottom_shape[0]*bottom_shape[2]*bottom_shape[3]; k<grid_dst_.count();k++){
    grid_dst_ptr[k] = grid_dst_b_ptr[k-bottom_shape[0]*bottom_shape[2]*bottom_shape[3]];
  }
  for (int l = 0; l<workspace_a.count();l++){
    workspace_ptr[l]=workspace_a_ptr[l];
  }
  for (int m = workspace_b.count(); m<2*workspace_b.count();m++){
    workspace_ptr[m]=workspace_b_ptr[m-workspace_b.count()];
  }

caffe_add(bottom_count, bottom_data, grid_dst_.cpu_data(), top_data);
caffe_div(bottom_count,top_data, workspace_.cpu_data(), top_data);
caffe_add_scalar(bottom_count, alpha, top_data);

}

template <typename Dtype>
void GridGeneratorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff  = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const vector<int> top_shape = top[0]->shape();
  const int top_count = top[0]->count();

  //workspace_
  Dtype workspace_term_a =  (top_shape[3]-1.0) / 2.0;
  Dtype workspace_term_b =  (top_shape[2]-1.0) / 2.0;

  Blob<Dtype> workspace_a;
  Blob<Dtype> workspace_b;
  workspace_a.Reshape(top_shape[0],1,top_shape[2], top_shape[3]);
  workspace_b.Reshape(top_shape[0],1,top_shape[2], top_shape[3]);
  Dtype *workspace_a_ptr = workspace_a.mutable_cpu_data();
  Dtype *workspace_b_ptr = workspace_b.mutable_cpu_data();

  for (int i = 0; i<workspace_a.count();i++) {
    workspace_a_ptr[i] = workspace_term_a;
    workspace_b_ptr[i] = workspace_term_b;
  }

  workspace_.Reshape(top_shape[0],2,top_shape[2], top_shape[3]);
  Dtype *workspace_ptr = workspace_.mutable_cpu_data();

  for (int l = 0; l<workspace_a.count();l++){
    workspace_ptr[l]=workspace_a_ptr[l];
  }
  for (int m = workspace_b.count(); m<2*workspace_b.count();m++){
    workspace_ptr[m]=workspace_b_ptr[m-workspace_b.count()];
  }

  caffe_div(top_count, top_diff, workspace_.cpu_data(), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(GridGeneratorLayer);
#endif

INSTANTIATE_CLASS(GridGeneratorLayer);
REGISTER_LAYER_CLASS(GridGenerator);

}  // namespace caffe
