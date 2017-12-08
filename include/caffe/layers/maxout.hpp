#ifndef CAFFE_MAXOUT_LAYER_HPP_
#define CAFFE_MAXOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {  
template <typename Dtype>  
class MaxoutLayer : public Layer<Dtype> {  
 public:  
  explicit MaxoutLayer(const LayerParameter& param)  
      : Layer<Dtype>(param) {}  
  virtual inline const char* type() const { return "Maxout"; }
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);  
  //virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
  //    vector<Blob<Dtype>*>* top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
  //    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);  

  int num_output_;  
  int num_;  
  int channels_;  
  int height_;  
  int width_;  
  int group_size_;  
  Blob<Dtype> max_idx_;  

};  

}
#endif
