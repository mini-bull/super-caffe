#include <cstdio>  
#include <vector>  

#include "caffe/util/im2col.hpp"   
#include "caffe/util/math_functions.hpp"  
#include "caffe/layers/maxout.hpp"  

namespace caffe {   
template <typename Dtype>  
void MaxoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,   
      vector<Blob<Dtype>*>& top) {
  const MaxoutParameter& maxout_param = this->layer_param_.maxout_param();
  CHECK(maxout_param.has_num_output())
      << "num_output should be specified.";   
}  

template <typename Dtype>  
void MaxoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  num_output_ = this->layer_param_.maxout_param().num_output();   
  CHECK_GT(num_output_, 0) << "output number cannot be zero.";  
  // bottom 
  num_ = bottom[0]->num();  
  channels_ = bottom[0]->channels();  
  height_ = bottom[0]->height();    
  width_ = bottom[0]->width();     

  // TODO: generalize to handle inputs of different shapes.    
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {    
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";    
    CHECK_EQ(channels_, bottom[bottom_id]->channels())    
        << "Inputs must have same channels.";    
    CHECK_EQ(height_, bottom[bottom_id]->height())    
        << "Inputs must have same height.";    
    CHECK_EQ(width_, bottom[bottom_id]->width())    
        << "Inputs must have same width.";    
  }  

  // Set the parameters, compute the group size
  CHECK_EQ(channels_ % num_output_, 0)   
      << "Number of channel should be multiples of output number.";   
  group_size_ = channels_ / num_output_;  

    top[0]->Reshape(num_, num_output_, height_, width_); 
    max_idx_.Reshape(num_, num_output_, height_, width_);
}


template <typename Dtype>   
void MaxoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    int featureSize = height_ * width_;  
    Dtype* mask = NULL;  
    mask = max_idx_.mutable_cpu_data();  
//printf("1.maxout_forward\n");  
    const int top_count = top[0]->count();  
    caffe_set(top_count, Dtype(0), mask);  
//printf("2.maxout_forward\n");  
    for (int i = 0; i < bottom.size(); ++i) {  
        const Dtype* bottom_data = bottom[i]->cpu_data();  
        Dtype* top_data = top[i]->mutable_cpu_data();    
        for (int n = 0; n < num_; n ++) {  
            for (int o = 0; o < num_output_; o ++) {  
                for (int g = 0; g < group_size_; g ++) {   
                    if (g == 0) {  
                        for (int h = 0; h < height_; h ++) { // á?2??-?·óDμ??ù?aà?  
                            for (int w = 0; w < width_; w ++) {  
                                int index = w + h * width_;  
                                top_data[index] = bottom_data[index];  
                                mask[index] = index;  
                            }  
                        }  
                    }  
                    else {  
                        for (int h = 0; h < height_; h ++) {  
                            for (int w = 0; w < width_; w ++) {  
                                int index0 = w + h * width_;  
                                int index1 = index0 + g * featureSize;  
                                if (top_data[index0] < bottom_data[index1]) {  
                                    top_data[index0] = bottom_data[index1];  
                                    mask[index0] = index1;  
                                }                                 
                            }  
                        }  
                    }  
                }  
                bottom_data += featureSize * group_size_;  
                top_data += featureSize;  
                mask += featureSize;  
            }  
        }  
    }    
}  



template <typename Dtype>  
void MaxoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
    if (!propagate_down[0]) {  
        return;  
    }  
    // const Dtype* top_diff = top[0]->cpu_diff();  
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);  
    const Dtype* mask = max_idx_.mutable_cpu_data();  
    int featureSize = height_ * width_;  
    for (int i = 0; i < top.size(); i ++) {  
        const Dtype* top_diff = top[i]->cpu_diff();  
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();    
        for (int n = 0; n < num_; n ++) { 
            for (int o = 0; o < num_output_; o ++) {  
                for (int h = 0; h < height_; h ++) { // á?2??-?·óDμ??ù?aà?  
                    for (int w = 0; w < width_; w ++) {  
                        int index = w + h * width_;  
                        int bottom_index = mask[index];  
                        bottom_diff[bottom_index] += top_diff[index];  
                    }  
                }  
                bottom_diff += featureSize * group_size_;  
                top_diff += featureSize;  
                mask += featureSize;  
            }  
        }  
    }  
} 

#ifdef CPU_ONLY
STUB_GPU(MaxoutLayer);
#endif
INSTANTIATE_CLASS(MaxoutLayer);
REGISTER_LAYER_CLASS(Maxout); 
}  // namespace caffe  
