#!/bin/bash


# ./data/cifar10/get_cifar10.sh
# ./examples/cifar10/create_cifar10.sh

sed 's/boost_weight_file:.*/boost_weight_file: \"\"/' cifar10_full.prototxt > net.prototxt
sed 's/output_boost_weight_file:.*/output_boost_weight_file: \"bw1_lmdb\"/' solver_template.prototxt > solver.prototxt
GLOG_logtostderr=1 ../../build/tools/caffe train -solver=solver.prototxt | tee output1.log

# sed 's/boost_weight_file:.*/boost_weight_file: \"bw1_lmdb\"/' cifar10_full.prototxt > net.prototxt
# sed 's/output_boost_weight_file:.*/output_boost_weight_file: \"bw2_lmdb\"/' solver_template.prototxt > solver.prototxt
# GLOG_logtostderr=1 ../../build/tools/caffe train -solver=solver_template.prototxt | tee output2.log