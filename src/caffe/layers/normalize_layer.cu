// Normalize neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layers/normalize_layer.hpp"

using namespace std;
namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Forward_cpu(bottom, top);
  if(norms_.size() == 0) norms_.resize(1000);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num = bottom[0]->shape()[0];
  const int num_channels = bottom[0]->shape()[1];
  assert(bottom[0]->shape()[3] == 1);
  assert(bottom[0]->shape()[2] == 1);
  //const Dtype* bottom_data_gpu = bottom[0]->gpu_data();

  for(int i = 0; i < num; ++i) {
    const Dtype* cur_bottom_data = bottom_data + i*num_channels;
    Dtype* cur_top_data = top_data + i*num_channels;

    caffe_gpu_dot(num_channels, cur_bottom_data, cur_bottom_data, &norms_[i]);
    norms_[i] = sqrt(1+norms_[i]);
    caffe_gpu_scale(num_channels, (Dtype)1./norms_[i], cur_bottom_data, cur_top_data);
  }


  /*
  const Dtype* a_diff = top[0]->cpu_data();
  Dtype s = 0.0;
  Dtype mx = -1e100;
  Dtype mn = +1e100;
    for (int i = 0; i < num_channels * num; ++i) {
      mx = max(mx, a_diff[i]);
      mn = min(mn, a_diff[i]);
      s += abs(a_diff[i]);
  }
  LOG(INFO) << "Normalize::Forward_gpu";
  SHOW(mn);
  SHOW(mx);
  */

  /*Dtype* bottom_data_cpu = bottom[0]->mutable_cpu_data();
  for(int i = 0; i < num; ++i) {
    Dtype mx = -1e100; Dtype mn = 1e100;
    for(int j = 0; j < num_channels; ++j) {
      mx = max(mx, bottom_data_cpu[i*num_channels+j]);
      mn = min(mn, bottom_data_cpu[i*num_channels+j]);
    }
    SHOW(mx);
    SHOW(mn);
    SHOW(norm_);
    SHOW(1/norm_/norm_/norm_);
  }
  */
}

/*
template <typename Dtype>
__global__ void normalize_kernel(const int n, Dtype norm, const Dtype* cur_top_diff,
    const Dtype* cur_top_data, Dtype* cur_bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    cur_bottom_diff[index] = cur_top_diff[index] * (1 - cur_top_data[index]*cur_top_data[index])/norm;
  }
}
*/

/*
template <typename Dtype>
__global__ void normalize_kernel(const int n, Dtype norm, const Dtype* cur_top_diff,
    const Dtype* cur_top_data, int k, Dtype* cur_bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    cur_bottom_diff[index] += cur_top_diff[k] * (- cur_top_data[index]*cur_top_data[k])/norm;
    cur_bottom_diff[index] += cur_top_diff[index]/norm/n;
  }
}
*/

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //Backward_cpu(top, propagate_down, bottom); return;
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int num = bottom[0]->shape()[0];
    const int num_channels = bottom[0]->shape()[1];

    caffe_gpu_memcpy(num*num_channels*sizeof(Dtype), top_diff, bottom_diff);

    for(int i = 0; i < num; ++i) {
      const Dtype* cur_top_data = top_data+i*num_channels;
      const Dtype* cur_top_diff = top_diff+i*num_channels;
      //const Dtype* cur_bottom_data = bottom_data+i*num_channels;
      Dtype* cur_bottom_diff = bottom_diff+i*num_channels;
      
      Dtype s;
      caffe_gpu_dot(num_channels, cur_top_data, cur_top_diff, &s);
      caffe_gpu_axpby(num_channels, -s/norms_[i], cur_top_data, 1/norms_[i], cur_bottom_diff);

      /*
      // NOLINT_NEXT_LINE(whitespace/operators)
      normalize_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_channels), CAFFE_CUDA_NUM_THREADS>>>(
      num_channels, norms_[i], cur_top_diff, cur_top_data, cur_bottom_diff);
      */

      // NOLINT_NEXT_LINE(whitespace/operators)
      //normalize_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_channels), CAFFE_CUDA_NUM_THREADS>>>(
      //num_channels, norms_[i], cur_top_diff, cur_top_data, k, cur_bottom_diff);

      /*caffe_gpu_mul(num_channels, cur_top_data, cur_top_data, cur_bottom_diff);
      caffe_gpu_add_scalar(num_channels, -(Dtype)1., cur_bottom_diff);
      caffe_gpu_scale(num_channels, -(Dtype)1./norm, cur_bottom_diff, cur_bottom_diff);
      caffe_gpu_mul(num_channels, cur_top_diff, cur_bottom_diff, cur_bottom_diff);
      */


      /*caffe_gpu_mul(num_channels, cur_bottom_data, cur_bottom_data, cur_bottom_diff);
      caffe_gpu_add_scalar(num_channels, -norm*norm, cur_bottom_diff);
      caffe_gpu_scale(num_channels, -(Dtype)1./norm/norm/norm, cur_bottom_diff, cur_bottom_diff);
      caffe_gpu_mul(num_channels, cur_top_diff, cur_bottom_diff, cur_bottom_diff);
      */
    }
    //SHOW(norm_sum/num);

    /*const Dtype* bottom_diff_cpu = bottom[0]->cpu_diff();
    cout << "GPU";
    for(int j = 0; j < 10; ++j)
      cout << bottom_diff_cpu[j] << " ";
    cout << endl;
    cout << endl;
    */

    /*Dtype* bottom_data_cpu = bottom[0]->mutable_cpu_data();
    for(int i = 0; i < num; ++i) {
    Dtype mx = -1e100; Dtype mn = 1e100;
    for(int j = 0; j < num_channels; ++j) {
      mx = max(mx, bottom_data_cpu[i*num_channels+j]);
      mn = min(mn, bottom_data_cpu[i*num_channels+j]);
    }
    SHOW(mx);
    SHOW(mn);
    SHOW(norm_);
    SHOW(1/norm_/norm_/norm_);
    
    ////jif(norm_ < 1e-5) {
    //SHOW(norm_);exit(0);
    //}
    
    }
    */

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe
