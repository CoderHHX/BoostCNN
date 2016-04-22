// Normalize neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layers/normalize_layer.hpp"

#define SHOW(x) LOG(ERROR) << #x << " " << x

using namespace std;
namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(norms_.size() == 0) norms_.resize(1000);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  //SHOW(bottom_data[0]);
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->shape()[0];
  const int num_channels = bottom[0]->shape()[1];
  assert(bottom[0]->shape()[3] == 1);
  assert(bottom[0]->shape()[2] == 1);

  /*Dtype mx = -1e100;
    Dtype mn = 1e100;

    for(int k = 0; k < num*num_channels; ++k)  {
    mx = max(mx, bottom_data[k]);
    mn = min(mn, bottom_data[k]);
    }
  //SHOW("data" << mn << " " << mx);
  */

  //SHOW(num);
  for(int i = 0; i < num; ++i) {
    Dtype* cur_top_data = top_data+i*num_channels;
    const Dtype* cur_bottom_data = bottom_data+i*num_channels;

    norms_[i] = caffe_cpu_dot(num_channels, cur_bottom_data, cur_bottom_data);
    //SHOW(norms_[i]);
    norms_[i] = sqrt(1+norms_[i]);

    //SHOW(norms_[i]);
    caffe_cpu_scale(num_channels, 1/norms_[i], cur_bottom_data, cur_top_data);
    
    //for(int j = 0; j < num_channels; ++j)
      //cur_top_data[j] = cur_bottom_data[j] / norms_[i];
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int num = bottom[0]->shape()[0];
    const int num_channels = bottom[0]->shape()[1];

    assert(bottom_data == top_data);
    assert(bottom_diff == top_diff);

    for(int i = 0; i < num; ++i) {
      const Dtype* cur_top_diff = top_diff+i*num_channels;
      const Dtype* cur_top_data = top_data+i*num_channels;
      //const Dtype* cur_bottom_data = bottom_data+i*num_channels;
      Dtype* cur_bottom_diff = bottom_diff+i*num_channels;
      //SHOW(norms_[i]);
      for(int j = 0; j < num_channels; ++j) {
        //cur_bottom_diff[j] = cur_top_diff[j] * (1 - cur_top_data[j]*cur_top_data[j])/norms_[i];
        cur_bottom_diff[j] = 0;

        for(int k = 0; k < num_channels; ++k) {
          cur_bottom_diff[j] += cur_top_diff[k] / norms_[i] * ((k==j) - cur_top_data[j]*cur_top_data[k]);
        }

      }
    }

       /* Dtype mx = -1e100;
        Dtype mn = 1e100;

        for(int k = 0; k < num*num_channels; ++k)  {
          mx = max(mx, bottom_diff[k]);
          mn = min(mn, bottom_diff[k]);
        }
        */
        //SHOW("diff" << mn << " " << mx);
    //SHOW(dum/num/num_channels);
    //SHOW(norm_sum/num);
    
    //cout << "CPU";
    //for(int j = 0; j < 10; ++j)
      //cout << bottom_diff[j] << " ";
    //cout << endl;

    /*cout << "CPU";
    for(int j = 0; j < 10; ++j)
      cout << top_data[j]*top_data[j]-1 << " ";
    cout << endl;

    cout << "CPU";
    for(int j = 0; j < 10; ++j)
      cout << bottom_data[j] << " ";
    cout << endl;
    */
    //SHOW(bottom_diff[0] << " " << bottom_diff[1]); 
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

REGISTER_LAYER_CLASS(Normalize);
INSTANTIATE_CLASS(NormalizeLayer);

}  // namespace caffe
