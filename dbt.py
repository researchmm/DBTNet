# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,missing-docstring
"""DBT, implemented in Gluon."""
from __future__ import division

__all__ = ['dbt']

import os
import math
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
import mxnet.gluon as gluon
import numpy as np

@mx.init.register
class Position4(mx.init.Initializer):
  def __init__(self):
    super(Position4, self).__init__()

  def cal_angle(self, position, hid_idx):
      return position * 1.0 / np.power(1.5, 2.0 * (hid_idx) / 32)

  def get_posi_angle_vec(self, position):
      return [self.cal_angle(position, hid_j) for hid_j in range(32)]

  def _init_weight(self, _, arr):

    n_position = 16
    sinusoid_table = np.array([self.get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    arr[:] = sinusoid_table.flat

@mx.init.register
class Position3(mx.init.Initializer):
  def __init__(self):
    super(Position3, self).__init__()

  def cal_angle(self, position, hid_idx):
      return position * 1.0 / np.power(1.5, 2.0 * (hid_idx) / 16)

  def get_posi_angle_vec(self, position):
      return [self.cal_angle(position, hid_j) for hid_j in range(16)]

  def _init_weight(self, _, arr):

    n_position = 16
    sinusoid_table = np.array([self.get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    arr[:] = sinusoid_table.flat

@mx.init.register
class Position2(mx.init.Initializer):
  def __init__(self):
    super(Position2, self).__init__()

  def cal_angle(self, position, hid_idx):
      return position * 1.0 / np.power(1.5, 2.0 * (hid_idx) / 16)

  def get_posi_angle_vec(self, position):
      return [self.cal_angle(position, hid_j) for hid_j in range(16)]

  def _init_weight(self, _, arr):

    n_position = 8
    sinusoid_table = np.array([self.get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    arr[:] = sinusoid_table.flat

class GB2HybridLayer(gluon.HybridBlock):
    def __init__(self, groups, per_group, width, myname):
        super(GB2HybridLayer, self).__init__()
        self.groups = groups
        self.per_group = per_group
        self.width = width
        self.myname = myname
        self.bias = self.params.get('bias',
                                      shape=(128),
                                      init=Position2(),
                                      differentiable=False)

        self.weight = self.params.get('weight',
                                      shape=(128,128),
                                      init=mx.initializer.Zero(),
                                      differentiable=False)
        self.body = nn.BatchNorm(gamma_initializer='zeros')

    def hybrid_forward(self, F, x, weight, bias):
        tmp = F.transpose(data = x, axes=(0,2,3,1))

        tmp = tmp.reshape((-1,128))
#
        mytmp = F.FullyConnected(tmp, num_hidden=128,weight = weight, bias=bias)
        tmp = tmp + mytmp

        tmp = tmp.reshape((-1,self.groups,self.per_group))
        tmp = F.transpose(data=tmp, axes=(0,2,1))
        tmp = tmp.astype('float32')
        tmp = F.tanh(F.BatchDot(data1 = tmp, data2 = tmp)/32).reshape((-1,self.width,self.width,self.per_group*self.per_group))
        tmp = F.contrib.BilinearResize2D(data = tmp, height = self.width, width = np.int(self.per_group*self.per_group*0.5))
        tmp = tmp.astype('float16')
#        tmp = tmp.reshape((-1,self.width,self.width,np.int(self.per_group*self.per_group*0.5)))
        return x + self.body(F.transpose(data = tmp, axes=(0,3,1,2)))

class GB3HybridLayer(gluon.HybridBlock):
    def __init__(self, groups, per_group, width, myname):
        super(GB3HybridLayer, self).__init__()
        self.groups = groups
        self.per_group = per_group
        self.width = width
        self.myname = myname
        self.bias = self.params.get('bias',
                                      shape=(256),
                                      init=Position3(),
                                      differentiable=False)

        self.weight = self.params.get('weight',
                                      shape=(256,256),
                                      init=mx.initializer.Zero(),
                                      differentiable=False)
        self.body = nn.BatchNorm(gamma_initializer='zeros')
    def hybrid_forward(self, F, x, weight, bias):
        tmp = F.transpose(data = x, axes=(0,2,3,1))

        tmp = tmp.reshape((-1,256))
#
        mytmp = F.FullyConnected(tmp, num_hidden=256,weight = weight, bias=bias)
        tmp = tmp + mytmp

        tmp = tmp.reshape((-1,self.groups,self.per_group))
        tmp = F.transpose(data=tmp, axes=(0,2,1))
        tmp = tmp.astype('float32')
        tmp = F.tanh(F.BatchDot(data1 = tmp, data2 = tmp)/32).reshape((-1,self.width,self.width,self.per_group*self.per_group))
        tmp = F.contrib.BilinearResize2D(data = tmp, height = self.width, width = np.int(self.per_group*self.per_group))
        tmp = tmp.astype('float16')
#        tmp = tmp.reshape((-1,self.width,self.width,np.int(self.per_group*self.per_group*0.5)))
        return x + self.body(F.transpose(data = tmp, axes=(0,3,1,2)))

class GB4HybridLayer(gluon.HybridBlock):
    def __init__(self, groups, per_group, width, myname):
        super(GB4HybridLayer, self).__init__()
        self.groups = groups
        self.per_group = per_group
        self.width = width
        self.myname = myname
        self.bias = self.params.get('bias',
                                      shape=(512),
                                      init=Position4(),
                                      differentiable=False)

        self.weight = self.params.get('weight',
                                      shape=(512,512),
                                      init=mx.initializer.Zero(),
                                      differentiable=False)
        self.body = nn.BatchNorm(gamma_initializer='zeros')

    def hybrid_forward(self, F, x, weight, bias):
        tmp = F.transpose(data = x, axes=(0,2,3,1))

        tmp = tmp.reshape((-1,512))
#
        mytmp = F.FullyConnected(tmp, num_hidden=512,weight = weight, bias=bias)
        tmp = tmp + mytmp

        tmp = tmp.reshape((-1,self.groups,self.per_group))
        tmp = F.transpose(data=tmp, axes=(0,2,1))
        tmp = tmp.astype('float32')
        tmp = F.tanh(F.BatchDot(data1 = tmp, data2 = tmp)/32).reshape((-1,self.width,self.width,self.per_group*self.per_group))
        tmp = F.contrib.BilinearResize2D(data = tmp, height = self.width, width = np.int(self.per_group*self.per_group*0.5))
        tmp = tmp.astype('float16')
#        tmp = tmp.reshape((-1,self.width,self.width,np.int(self.per_group*self.per_group*0.5)))
        return x + self.body(F.transpose(data = tmp, axes=(0,3,1,2)))


class GroupConv(nn.Conv2D):
    def __init__(self, *args, **kwargs):
        self.width=kwargs['width']
        del kwargs['width']
        super(GroupConv, self).__init__(*args, **kwargs)
        self.body = nn.BatchNorm()
    def hybrid_forward(self, F, x, weight, bias=None):
#        weight = weight*10000
        groups = 16
        channels = self._channels
        width = self.width
        act = super(GroupConv, self).hybrid_forward(F, x, weight, bias)
        act = self.body(act)
        act = F.Activation(data=act,act_type ='relu')
        tmp = act+0.001
        tmp = F.L2Normalization(tmp.reshape((-1,width*width)), mode='instance')
        tmp = F.transpose(tmp.reshape((-1,channels,width*width)),axes=(1,0,2)).reshape((channels,-1))
        co = F.dot(tmp,tmp,False,True).reshape((1,channels*channels))/128
#        tmp = tmp.reshape((-1,channels,width*width)).astype('float32')
#        co = F.BatchDot(tmp,tmp).astype('float16').reshape((128,channels*channels))
        gt = F.tile(F.ones(groups).diag().reshape((1, 1, groups, groups)),(1, np.int((channels/groups)*(channels/groups)), 1, 1))
        gt = F.depth_to_space(gt, np.int(channels/groups)).astype('float16').reshape((1,channels*channels))
        loss = F.tile(F.sum((co-gt)*(co-gt)*0.001,axis=1),(48))/((channels/512.0)*(channels/512.0))
#        loss = (co-gt)*(co-gt)
        self.loss = loss
        return act


class GroupConv2(nn.Conv2D):
    def __init__(self, *args, **kwargs):
        self.width=kwargs['width']
        del kwargs['width']
        super(GroupConv2, self).__init__(*args, **kwargs)
        self.body = nn.BatchNorm()
    def hybrid_forward(self, F, x, weight, bias=None):
#        weight = weight*10000
        groups = 32
        channels = self._channels
        width = self.width
        act = super(GroupConv2, self).hybrid_forward(F, x, weight, bias)
        act = self.body(act)
        act = F.Activation(data=act,act_type ='relu')
        tmp = act+0.001
        tmp = F.L2Normalization(tmp.reshape((-1,width*width)), mode='instance')
        tmp = F.transpose(tmp.reshape((-1,channels,width*width)),axes=(1,0,2)).reshape((channels,-1))
        co = F.dot(tmp,tmp,False,True).reshape((1,channels*channels))/128
#        tmp = tmp.reshape((-1,channels,width*width)).astype('float32')
#        co = F.BatchDot(tmp,tmp).astype('float16').reshape((128,channels*channels))
        gt = F.tile(F.ones(groups).diag().reshape((1, 1, groups, groups)),(1, np.int((channels/groups)*(channels/groups)), 1, 1))
        gt = F.depth_to_space(gt, np.int(channels/groups)).astype('float16').reshape((1,channels*channels))
        loss = F.tile(F.sum((co-gt)*(co-gt)*0.001,axis=1),(48))/((channels/512.0)*(channels/512.0))
#        loss = (co-gt)*(co-gt)
        self.loss = loss
        return act


class Block(HybridBlock):
    def __init__(self, channels, stride,
                 downsample=False, last_gamma=False, use_se=False, **kwargs):
        super(Block, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                                groups=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels*4, kernel_size=1, use_bias=False))
        if last_gamma:
            self.body.add(nn.BatchNorm())
        else:
            self.body.add(nn.BatchNorm(gamma_initializer='zeros'))



        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Conv2D(channels // 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Conv2D(channels * 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)
        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
#        self.loss = self.gconv.loss
        return x


class Block3(HybridBlock):
    def __init__(self, channels, stride, myname,
                 downsample=False, last_gamma=False, use_se=False, **kwargs):
        super(Block3, self).__init__(**kwargs)
        width = np.int(56*64/channels*stride)*2

        self.body = nn.HybridSequential(prefix='')
        gconv = GroupConv(channels, kernel_size=1, use_bias=False, width=width)
        self.gconv = gconv
        self.body.add(gconv)
        self.body.add(GB3HybridLayer(16, np.int(channels/16), width, myname))
        self.body.add(nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                                groups=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))


        self.body.add(nn.Conv2D(channels*4, kernel_size=1, use_bias=False))
        if last_gamma:
            self.body.add(nn.BatchNorm())
        else:
            self.body.add(nn.BatchNorm(gamma_initializer='zeros'))



        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Conv2D(channels // 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Conv2D(channels * 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)
        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        self.loss = self.gconv.loss
        return x

class Block4(HybridBlock):
    def __init__(self, channels, stride, myname,
                 downsample=False, last_gamma=False, use_se=False, **kwargs):
        super(Block4, self).__init__(**kwargs)
        width = np.int(56*64/channels*stride)*2

        self.body = nn.HybridSequential(prefix='')
        gconv = GroupConv(channels, kernel_size=1, use_bias=False, width=width)
        self.gconv = gconv
        self.body.add(gconv)
        self.body.add(GB4HybridLayer(16, np.int(channels/16), width, myname))
        self.body.add(nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                                groups=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))


        self.body.add(nn.Conv2D(channels*4, kernel_size=1, use_bias=False))
        if last_gamma:
            self.body.add(nn.BatchNorm())
        else:
            self.body.add(nn.BatchNorm(gamma_initializer='zeros'))



        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Conv2D(channels // 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Conv2D(channels * 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)
        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        self.loss = self.gconv.loss
        return x
# Nets
class DBTNet(HybridBlock):
    def __init__(self, layers, classes=1000, last_gamma=False, **kwargs):
        super(DBTNet, self).__init__(**kwargs)
        channels = 64
        self.gconvs = nn.HybridSequential(prefix='')
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features1 = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))

            self.features.add(nn.Conv2D(channels, 7, 2, 3, use_bias=False))

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(channels, num_layer, stride,
                                                   last_gamma, False, i+1))
                channels *= 2
            gconv = GroupConv2(2048, kernel_size=1, use_bias=False, width=14)
            self.gconvs.add(gconv)
            self.features.add(gconv)
            self.features.add(GB2HybridLayer(32, np.int(2048/32), 14, 'gb'))
            self.features1.add(nn.GlobalAvgPool2D())
            self.features1.add(nn.Flatten())

        self.myoutput = nn.HybridSequential(prefix='new')
        with self.myoutput.name_scope():
            self.myoutput.add(nn.Conv2D(classes, kernel_size=1, padding=0, use_bias=True))

    def _make_layer(self, channels, num_layers, stride, last_gamma, use_se, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            if stage_index<3:
                myblock = Block(channels, stride, True, last_gamma=last_gamma, use_se=use_se, prefix='')
                layer.add(myblock)
                for i in range(num_layers-1):
                    myblock = Block(channels, 1, False, last_gamma=last_gamma, use_se=use_se, prefix='')
                    layer.add(myblock)
            elif stage_index==3:
                myblock = Block3(channels, stride, 0, True, last_gamma=last_gamma, use_se=use_se, prefix='')
                layer.add(myblock)
#                with self.gconvs.name_scope:
                self.gconvs.add(myblock.gconv)
                for i in range(num_layers-1):
                    myblock = Block3(channels, 1, i+1, False, last_gamma=last_gamma, use_se=use_se, prefix='')
                    layer.add(myblock)
#                with self.gconvs.name_scope:
                    self.gconvs.add(myblock.gconv)

            elif stage_index==4:
                myblock = Block4(channels, stride, 0, True, last_gamma=last_gamma, use_se=use_se, prefix='')
                layer.add(myblock)
#                with self.gconvs.name_scope:
                self.gconvs.add(myblock.gconv)
                for i in range(num_layers-1):
                    myblock = Block4(channels,1, i+1, False, last_gamma=last_gamma, use_se=use_se, prefix='')
                    layer.add(myblock)
#                with self.gconvs.name_scope:
                    self.gconvs.add(myblock.gconv)
               	
        return layer

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.myoutput(x)
        x = self.features1(x)
        cnt = 0
        for gconv in self.gconvs._children.values():
            if cnt == 0:
                loss = gconv.loss
            else:
                loss = loss + gconv.loss
            cnt = cnt+1
        loss = loss/cnt
        return x, loss


# Specification
resnet_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}


# Constructor
def get_dbt(num_layers, pretrained=False, ctx=cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnext_spec.keys()))
    layers = resnet_spec[num_layers]
    net = DBTNet(layers, **kwargs)
    return net

def dbt(**kwargs):
    return get_dbt(50, 32, 4, **kwargs)

