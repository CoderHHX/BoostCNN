#include <vector>
#include <limits.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bilinear_layer.hpp"

namespace caffe {

template <typename Dtype>
void BilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->shape()[0], bottom[1]->shape()[0]);
  CHECK_EQ(bottom[0]->shape()[2], bottom[1]->shape()[2]);
  CHECK_EQ(bottom[0]->shape()[3], bottom[1]->shape()[3]);

  // const int axis = bottom[0]->CanonicalAxisIndex(
  //     this->layer_param_.inner_product_param().axis());
  // // Dimensions starting from "axis" are "flattened" into a single
  // // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // // and axis == 1, N inner products with dimension CHW are performed.
  // K_ = bottom[0]->count(axis);
  // // Check if we need to set up the weights
  // if (this->blobs_.size() > 0) {
  //   LOG(INFO) << "Skipping parameter initialization";
  // } else {
  //   if (bias_term_) {
  //     this->blobs_.resize(2);
  //   } else {
  //     this->blobs_.resize(1);
  //   }
  //   // Intialize the weight
  //   vector<int> weight_shape(2);
  //   weight_shape[0] = N_;
  //   weight_shape[1] = K_;
  //   this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  //   // fill the weights
  //   shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
  //       this->layer_param_.inner_product_param().weight_filler()));
  //   weight_filler->Fill(this->blobs_[0].get());
  //   // If necessary, intiialize and fill the bias term
  //   if (bias_term_) {
  //     vector<int> bias_shape(1, N_);
  //     this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
  //     shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
  //         this->layer_param_.inner_product_param().bias_filler()));
  //     bias_filler->Fill(this->blobs_[1].get());
  //   }
  // }  // parameter initialization
  // this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // // Figure out the dimensions
  // const int axis = bottom[0]->CanonicalAxisIndex(
  //     this->layer_param_.inner_product_param().axis());
  // const int new_K = bottom[0]->count(axis);
  // CHECK_EQ(K_, new_K)
  //     << "Input size incompatible with inner product parameters.";
  // // The first "axis" dimensions are independent inner products; the total
  // // number of these is M_, the product over these dimensions.
  // M_ = bottom[0]->count(0, axis);
  // // The top shape will be the bottom shape with the flattened axes dropped,
  // // and replaced by a single axis with dimension num_output (N_).
  // vector<int> top_shape = bottom[0]->shape();
  // top_shape.resize(axis + 1);
  // top_shape[axis] = N_;
  // top[0]->Reshape(top_shape);
  // // Set up the bias multiplier
  // if (bias_term_) {
  //   vector<int> bias_shape(1, M_);
  //   bias_multiplier_.Reshape(bias_shape);
  //   caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  // }

  //LOG(INFO) << "Reshape is called ================================";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = bottom[0]->shape()[1] * bottom[1]->shape()[1];
  top_shape[2] = 1;
  top_shape[3] = 1;
  top[0]->Reshape(top_shape);
  //LOG(INFO) << top_shape[0] << " " << top_shape[1] << " " << top_shape[2] << " " << top_shape[3];
}

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = (Dtype*)top[0]->mutable_cpu_data();
  const Dtype* a_data = (Dtype*)bottom[0]->cpu_data();
  const Dtype* b_data = (Dtype*)bottom[1]->cpu_data();
  
  int batch_size = bottom[0]->shape()[0];
  int num_channels1 = bottom[0]->shape()[1];
  int num_channels2 = bottom[1]->shape()[1];
  int hw = bottom[0]->shape()[2] * bottom[0]->shape()[3];

  /*for(int n = 0; n < batch_size; ++n)
    for(int c1 = 0; c1 < num_channels1; ++c1)
      for(int c2 = 0; c2 < num_channels2; ++c2)
        top_data[n*num_channels1*num_channels2+c1*num_channels2+c2] = 
          caffe_cpu_dot(hw, a_data+n*hw*num_channels1+c1*hw, b_data+n*hw*num_channels2+c2*hw) / hw;
          */
  
  for(int n = 0; n < batch_size; ++n)
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_channels1, num_channels2, hw,
                   Dtype(1.0), a_data+n*hw*num_channels1,
                   b_data+n*hw*num_channels2, Dtype(0.0),
                   top_data+n*num_channels1*num_channels2);
/*

  Dtype mn = 1e20;
  Dtype mx = -1e20;
  Dtype av = 0;
  for(int i = 0; i < bottom[0]->count(); ++i) {
    Dtype x = a_data[i];
    mn = std::min(mn, x);
    mx = std::max(mx, x);
    av += x / top[0]->count();
  }

  SHOW(mn);
  SHOW(mx);
  SHOW(av);
  */

}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  int batch_size = top[0]->shape()[0];
  int num_channels1 = bottom[0]->shape()[1];
  int num_channels2 = bottom[1]->shape()[1];
  int hw = bottom[0]->shape()[2] * bottom[0]->shape()[3];

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* a_data = bottom[0]->cpu_data();
  const Dtype* b_data = bottom[1]->cpu_data();

  Dtype* a_diff = bottom[0]->mutable_cpu_diff();
  Dtype* b_diff = bottom[1]->mutable_cpu_diff();

  if(propagate_down[0])
    caffe_set(batch_size*num_channels1*hw, (Dtype)0.0, a_diff);

  if(propagate_down[1])
    caffe_set(batch_size*num_channels2*hw, (Dtype)0.0, b_diff);


        /*for(int c = 0; c < num_channels1; ++c)
          for(int i = 0; i < hw; ++i)
            a_diff[n*num_channels1*hw + c*hw + i] = 0;
            */

        /*for(int c1 = 0; c1 < num_channels1; ++c1)
        for(int c2 = 0; c2 < num_channels2; ++c2)
          for(int i = 0; i < hw; ++i)
            a_diff[n*num_channels1*hw + c1*hw + i] += b_data[n*num_channels2*hw+c2*hw+i] * top_diff[n*num_channels1*num_channels2 + c2*num_channels2+c1];
            */
        /*for(int c1 = 0; c1 < num_channels1; ++c1)
        for(int c2 = 0; c2 < num_channels2; ++c2)
          for(int i = 0; i < hw; ++i)
            b_diff[n*num_channels1*hw + c2*hw + i] += a_data[n*num_channels2*hw+c1*hw+i] * top_diff[n*num_channels1*num_channels2 + c2*num_channels2+c1];
            */

  for(int n = 0; n < batch_size; ++n) {
    if(propagate_down[0])
      caffe_cpu_gemm(CblasTrans, CblasNoTrans, num_channels1, hw, num_channels2,
                     Dtype(1.0), top_diff+n*num_channels1*num_channels2,
                     b_data+n*num_channels2*hw, Dtype(1.0),
                     a_diff+n*num_channels1*hw);

    if(propagate_down[1])
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_channels2, hw, num_channels1,
                     Dtype(1.0), top_diff+n*num_channels1*num_channels2,
                     a_data+n*num_channels1*hw, Dtype(1.0),
                     b_diff+n*num_channels2*hw);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BilinearLayer);
#endif

INSTANTIATE_CLASS(BilinearLayer);
REGISTER_LAYER_CLASS(Bilinear);

}  // namespace caffe
