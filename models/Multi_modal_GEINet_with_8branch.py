# coding:utf-8

import numpy as np
import os
import glob
from PIL import Image
import cv2
import chainer
import cupy as cp
from chainer import Serializer, training, reporter, iterators, Function, cuda, initializers
from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer.training import extensions
from chainer import serializers
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.functions.evaluation import accuracy
import six

from sklearn.neighbors import KNeighborsClassifier


class Multi_modal_GEINet(Chain):
    def __init__(self):
        super(Multi_modal_GEINet, self).__init__()
        self.W = initializers.HeNormal()
        self.b = initializers.Constant(fill_value=0)
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=2, in_channels=3, out_channels=18, ksize=7, stride=1, initialW=self.W,
                                         initial_bias=self.b)
            self.conv2 = L.ConvolutionND(ndim=2, in_channels=18, out_channels=45, ksize=5, stride=1, pad=(2, 2),
                                         initialW=self.W, initial_bias=self.b)
            # 6branch以降はfc3のパラメータ数を固定
            self.fc3 = L.Linear(None, 5120, initialW=self.W, initial_bias=self.b)
            self.fc4 = L.Linear(None, 956, initialW=self.W, initial_bias=self.b)

    def GEIBlock(self, data):
        h1 = F.relu(self.conv1(data))
        h2 = F.max_pooling_2d(h1, stride=2, ksize=2)
        h3 = F.local_response_normalization(h2, k=1, n=5, alpha=0.00002, beta=0.75)
        h4 = F.relu(self.conv2(h3))
        h5 = F.max_pooling_2d(h4, stride=2, ksize=3)
        h6 = F.local_response_normalization(h5, k=1, n=5, alpha=0.00002, beta=0.75)

        return h6

    def __call__(self, a, b, c, d, e, f, g, h):
        root1 = self.GEIBlock(a)
        root2 = self.GEIBlock(b)
        root3 = self.GEIBlock(c)
        root4 = self.GEIBlock(d)
        root5 = self.GEIBlock(e)
        root6 = self.GEIBlock(f)
        root7 = self.GEIBlock(g)
        root8 = self.GEIBlock(h)

        h6 = F.concat((root1, root2), axis=1)
        h6 = F.concat((h6, root3), axis=1)
        h6 = F.concat((h6, root4), axis=1)
        h6 = F.concat((h6, root5), axis=1)
        h6 = F.concat((h6, root6), axis=1)
        h6 = F.concat((h6, root7), axis=1)
        h7 = F.concat((h6, root8), axis=1)

        h8 = F.dropout(F.relu(self.fc3(h7)), ratio=0.5)
        h9 = self.fc4(h8)
        # h9 = F.softmax(h8)
        return h9


