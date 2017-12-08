#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/layers/bilinearsampler_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void BilinearSamplerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 // const BilinearSamplerParameter& param = this->layer_param_.BilinearSampler_param();
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void BilinearSamplerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* grid_dst = bottom[0]->cpu_data();
  const Dtype* data = bottom[1]->cpu_data(); 
  Dtype* top_data = top[0]->mutable_cpu_data();

  const vector<int> top_shape = top[0]->shape();
  const vector<int> data_shape = bottom[1]->shape();

  int o_n = top_shape[0], o_c = top_shape[1], o_h = top_shape[2], o_w = top_shape[3];
  int i_c = data_shape[1], i_h = data_shape[2], i_w = data_shape[3];
  for (int n = 0; n < o_n; ++n) {
    for (int c = 0; c < o_c; ++c) {
      for (int h = 0; h < o_h; ++h) {
        for (int w = 0; w < o_w; ++w) {
          int out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
          int grid_index = n * o_h * o_w * 2 + h * o_w + w;
          Dtype y_real = (grid_dst[grid_index + o_h * o_w] + 1)* (i_h - 1) / 2;
          Dtype x_real = (grid_dst[grid_index] + 1) * (i_w - 1) / 2;
          int top_left_y = floor(y_real);
          int top_left_x = floor(x_real);
          Dtype top_left_y_w = 1.0 - (y_real - top_left_y);
          Dtype top_left_x_w = 1.0 - (x_real - top_left_x);
          int data_index = n * i_c * i_h * i_w + c * i_h * i_w +
            top_left_y * i_w + top_left_x;
          Dtype top_left_v = 0;
          Dtype top_right_v = 0;
          Dtype bottom_left_v = 0;
          Dtype bottom_right_v = 0;
          if (top_left_x>=0 && top_left_x<=i_w-1 && top_left_y>=0 && top_left_y<=i_h-1) {
            top_left_v = data[data_index];
          }
          if (top_left_x + 1>=0 && top_left_x + 1<=i_w-1 && top_left_y>=0 && top_left_y<=i_h-1) {
            top_right_v = data[data_index+1];
          }
          if (top_left_x>=0 && top_left_x<=i_w-1 && top_left_y + 1>=0 && top_left_y + 1<=i_h-1) {
            bottom_left_v = data[data_index + i_w];
          }
          if (top_left_x + 1>=0 && top_left_x + 1<=i_w-1 && top_left_y + 1>=0 && top_left_y + 1<=i_h-1) {
            bottom_right_v = data[data_index + i_w + 1];
          }
          top_data[out_index] = top_left_v * top_left_y_w * top_left_x_w +
                              top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                              bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                              bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
        }
      }
    }
  }
}

template <typename Dtype>
void BilinearSamplerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();//input diff
  Dtype* bottom_grid_diff = bottom[0]->mutable_cpu_diff();//grid(optical flow) grad
  Dtype* bottom_data_diff = bottom[1]->mutable_cpu_diff();//object and segment grad
  const Dtype* grid_dst = bottom[0]->cpu_data();
  const Dtype* data = bottom[1]->cpu_data(); 

  const vector<int> top_shape = top[0]->shape();
  const vector<int> bottom_shape = bottom[1]->shape();

  int o_n = bottom_shape[0], o_c = bottom_shape[1], o_h = bottom_shape[2], o_w = bottom_shape[3];
  int i_c = top_shape[1], i_h = top_shape[2], i_w = top_shape[3];
  for (int n = 0; n < o_n; ++n) {
     for (int h = 0; h < o_h; ++h) {
        for (int w = 0; w < o_w; ++w) {
          Dtype top_left_y_gw = 0.0;
          Dtype top_left_x_gw = 0.0;
          int grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
          Dtype y_real = (grid_dst[grid_src_index + o_h * o_w] + 1) * (i_h - 1) / 2;
          Dtype x_real = (grid_dst[grid_src_index] + 1) * (i_w - 1) / 2;
          int top_left_y = floor(y_real);
          int top_left_x = floor(x_real);
          Dtype top_left_y_w = 1.0 - (y_real - top_left_y);
          Dtype top_left_x_w = 1.0 - (x_real - top_left_x);
          for (int c = 0; c < o_c; ++c) {
            int grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
            int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w
                                  + top_left_x;
            // calc 4 vertex value in input data
            Dtype top_left_v = 0;
            Dtype top_right_v = 0;
            Dtype bottom_left_v = 0;
            Dtype bottom_right_v = 0;
            // calc input grad
            if (top_left_x>=0 && top_left_x<=i_w-1 && top_left_y>=0 && top_left_y<=i_h-1) {
              bottom_data_diff[data_index] += top_diff[grad_index] * top_left_y_w * top_left_x_w;
              top_left_v = data[data_index];
            }
            if (top_left_x + 1>=0 && top_left_x + 1<=i_w-1 && top_left_y>=0 && top_left_y<=i_h-1) {
              bottom_data_diff[data_index+1] += top_diff[grad_index] * top_left_y_w
                                              * (1.0 - top_left_x_w);
              top_right_v = data[data_index + 1];
            }
            if (top_left_x>=0 && top_left_x<=i_w-1 && top_left_y + 1>=0 && top_left_y + 1<=i_h-1) {
              bottom_data_diff[data_index + i_w] += top_diff[grad_index] * (1.0 - top_left_y_w)
                                              * top_left_x_w;
              bottom_left_v = data[data_index + i_w];
            }
            if (top_left_x + 1>=0 && top_left_x + 1<=i_w-1 && top_left_y + 1>=0 && top_left_y + 1<=i_h-1) {
              bottom_data_diff[data_index+ i_w + 1] += top_diff[grad_index] * (1.0 - top_left_y_w)
                                                  * (1.0 - top_left_x_w);
              bottom_right_v = data[data_index + i_w + 1];
            }
            // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
            top_left_y_gw -= top_diff[grad_index] * (top_right_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_x_w);
            top_left_x_gw -= top_diff[grad_index] * (bottom_left_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_y_w);
          }
          // calc grad of grid
          bottom_grid_diff[grid_src_index + o_h * o_w] += top_left_y_gw * (i_h - 1) / 2;
          bottom_grid_diff[grid_src_index] += top_left_x_gw * (i_w - 1) / 2;
        }
      }
    }
  //NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(BilinearSamplerLayer);
#endif

INSTANTIATE_CLASS(BilinearSamplerLayer);
REGISTER_LAYER_CLASS(BilinearSampler);

} // namespace caffe


