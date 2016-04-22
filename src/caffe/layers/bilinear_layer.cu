#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bilinear_layer.hpp"

namespace caffe {

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
/*
  Dtype a[12] = {6.0, 3.0, 6.0, 3.0, 0.0, 6.0, 5.0, 8.0, 5.0, 6.0, 2.0, 2.0};
  Dtype b[6] = {9.0, 8.0, 2.0, 10.0, 10.0, 0.0};
  Dtype c[8];
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 4, 2, 3,
                 Dtype(1.0),
                 a, b, Dtype(0.0),
                 c);
  for(int i = 0; i < 8; ++i)
    SHOW(c[i]);
  exit(0);
  */



  //Forward_gpu(bottom, top);
  //return;
  Dtype* top_data = (Dtype*)top[0]->mutable_gpu_data();
  const Dtype* a_data = (Dtype*)bottom[0]->gpu_data();
  const Dtype* b_data = (Dtype*)bottom[1]->gpu_data();
  
  int batch_size = bottom[0]->shape()[0];
  int num_channels1 = bottom[0]->shape()[1];
  int num_channels2 = bottom[1]->shape()[1];
  int hw = bottom[0]->shape()[2] * bottom[0]->shape()[3];

  assert(a_data == b_data);



  /*SHOW(batch_size);
  SHOW(num_channels1);
  SHOW(hw);
  SHOW(bottom[0]->num());
  SHOW(bottom[0]->height());
  SHOW(bottom[0]->width());
  SHOW(bottom[0]->channels());
  */

  for(int n = 0; n < batch_size; ++n) {
//    SHOW(n);
//    for(int c1 = 0; c1 < num_channels1; ++c1)
//      for(int c2 = 0; c2 < num_channels2; ++c2) {
//        //SHOW(n << " " << c1 << " " << c2);
//        caffe_gpu_dot(hw,
//                      a_data+n*hw*num_channels1+c1*hw,
//                      b_data+n*hw*num_channels2+c2*hw,
//                      &top_data[n*num_channels1*num_channels2+c1*num_channels2+c2]); 
//        
//      }

//    SHOW(hw);
//    SHOW(num_channels1);
//    SHOW(num_channels2);

//  caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_channels1, num_channels2, hw,
//                 Dtype(1.0), a_data+n*hw*num_channels1, x,
//                 Dtype(0.0),
//                 x);//top_data+n*num_channels1*num_channels2);


    caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_channels1, num_channels2, hw,
                   Dtype(1.0), a_data+n*hw*num_channels1,
                   b_data+n*hw*num_channels2, Dtype(0.0),
                   top_data+n*num_channels1*num_channels2);
  }


  /*const Dtype* a_data_gpu = (Dtype*)bottom[0]->gpu_data();
  Dtype mx = -1e100;
  Dtype mn = +1e100;
  SHOW(num_channels1 * hw * batch_size);
  for (int i = 0; i < num_channels1 * hw * batch_size; ++i) {
      if (a_data_gpu[i] > mx) {mx = a_data_gpu[i];}
      if (a_data_gpu[i] < mn) {mn = a_data_gpu[i];}
  }
  LOG(INFO) << "Forward_gpu";
  SHOW(mn);
  SHOW(mx);
  */

//  SHOW(bottom[0]->gpu_data()[bottom[0]->count()-1]);
//  SHOW(bottom[1]->gpu_data()[bottom[1]->count()-1]);
//  SHOW(top[0]->gpu_data()[top[0]->count()-1]);
//
//  for(int i = 0; i < bottom[0]->count(); ++i)
//    if(isnan(bottom[0]->gpu_data()[i])) {
//      SHOW(i);
//      exit(0);
//    }
//
//  for(int i = 0; i < top[0]->count(); ++i)
//    if(isnan(top[0]->gpu_data()[i])) {
//      SHOW(i);
//      exit(0);
//    }

}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  //LOG(ERROR) << "NOT I66MPLEMENTED";
  //exit(0);
  int batch_size = top[0]->shape()[0];
  int num_channels1 = bottom[0]->shape()[1];
  int num_channels2 = bottom[1]->shape()[1];
  int hw = bottom[0]->shape()[2] * bottom[0]->shape()[3];

  //LOG(ERROR) << "$$$$$$$$$$$$" << " " << batch_size << " " << num_channels1 << " " << num_channels2 << " " << hw;

  const Dtype* top_data_gpu = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* a_data = bottom[0]->gpu_data();
  const Dtype* b_data = bottom[1]->gpu_data();
  assert(a_data == b_data);
  assert(a_diff == b_diff);

  Dtype* a_diff = bottom[0]->mutable_gpu_diff();
  Dtype* b_diff = bottom[1]->mutable_gpu_diff();

  if(propagate_down[0])
    caffe_gpu_set(num_channels1 * hw * batch_size, (Dtype)0.0, a_diff);
  if(propagate_down[1])
    caffe_gpu_set(num_channels1 * hw * batch_size, (Dtype)0.0, b_diff);

    for(int n = 0; n < batch_size; ++n) {
      if (propagate_down[0])
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_channels1, hw, num_channels2,
                       Dtype(1.0), top_diff+n*num_channels1*num_channels2,
                       b_data+n*num_channels2*hw, Dtype(1.0),
                       a_diff+n*num_channels1*hw);

        
      if( propagate_down[1])
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_channels2, hw, num_channels1,
                       Dtype(1.0), top_diff+n*num_channels1*num_channels2,
                       a_data+n*num_channels1*hw, Dtype(1.0),
                       b_diff+n*num_channels2*hw);
                       

      /*
      if (propagate_down[1])
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_channels1, hw, num_channels2,
                       Dtype(1.0), top_diff+n*num_channels1*num_channels2,
                       a_data+n*hw*num_channels2, Dtype(0.0),
                       b_diff+n*num_channels1*hw);
                       */
           
        
        /*
        for(int i = 0; i < hw; ++i)
        for(int c1 = 0; c1 < num_channels1; ++c1)
        for(int c2 = 0; c2 < num_channels2; ++c2)
          a_diff[n*num_channels1*hw + c1 * hw + i] += 2*top_diff[n*num_channels1*num_channels2+c1*num_channels2+c2] * b_data[n*num_channels2*hw + c2*hw + i];
 
 */

    /*a_diff[c1, i] = sum_c2 {top_diff[c1, c2] * b[c2,i] }
    top[c1,c2]  = sum_i {a[c1,i] * b[c2,i]}
    d top[c1,c2] / d a[c1, i] = b[c2, i]
    d L / d a[c1,i] = sum_xy {(d L / d top[x,y]) * (d top[x,y] / d a[c1, i])}
    d L / d a[c1,i] = sum_y {(d L / d top[c1,y]) * (d top[c1,y] / d a[c1, i])}
    }
    */
  }

  /*Dtype s = 0.0;
  Dtype mx = -1e100;
  Dtype mn = +1e100;
    for (int i = 0; i < num_channels1 * hw * batch_size; ++i) {
      mx = max(mx, a_diff[i]);
      mn = min(mn, a_diff[i]);
      s += abs(b_diff[i]);
  }
  LOG(INFO) << "Backward_gpu";
  SHOW(mn);
  SHOW(mx);
  */


  /*
    caffe_gpu_abs(num_channels1 * hw * batch_size, b_diff, b_diff);
    caffe_gpu_powx(num_channels1 * hw * batch_size, b_diff, (Dtype)0.5, b_diff);
    Dtype s;
    caffe_gpu_dot(num_channels1 * hw * batch_size, b_diff, b_diff, &s);
    SHOW(s/ (num_channels1 * hw * batch_size));
    */


    //caffe_gpu_scal(num_channels1 * hw * batch_size, (Dtype)0.0, a_diff);
    //caffe_gpu_scal(num_channels1 * hw * batch_size, (Dtype)0.0, b_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearLayer);

}  // namespace caffe
