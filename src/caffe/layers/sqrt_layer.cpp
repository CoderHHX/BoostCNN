// Sqrt neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layers/sqrt_layer.hpp"

#define EPS ((Dtype) 1e-10)
namespace caffe {

template <typename Dtype>
void SqrtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] > 0)
      top_data[i] = sqrt(bottom_data[i]);
    else
      top_data[i] = -sqrt(bottom_data[i]);
  }
}

template <typename Dtype>
void SqrtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = 0.5 * top_diff[i] / (EPS+abs(top_data[i]));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SqrtLayer);
#endif

REGISTER_LAYER_CLASS(Sqrt);
INSTANTIATE_CLASS(SqrtLayer);

}  // namespace caffe
