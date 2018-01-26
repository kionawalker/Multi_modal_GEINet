# _*_ coding:utf-8 _*_

import numpy as np
import os
import glob
from PIL import Image
import cv2
import chainer
import cupy as cp
from chainer import Serializer, training, reporter, iterators, Function, cuda ,initializers
from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer.training import extensions
from chainer import serializers
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.functions.evaluation import accuracy
import six

import sys
sys.path.append('/home/common-ns/PycharmProjects/Prepare')
from tools import load_image
sys.path.append('/home/common-ns/PycharmProjects/models')
import Multi_modal_GEINet
from sklearn.neighbors import KNeighborsClassifier


class Multi_modal_Updater(training.StandardUpdater):

    def __init__(self, Mymodel, iterator1, iterator2, optimizer, converter=convert.concat_examples, device=None,
                 loss_func=None):
        super(Multi_modal_Updater, self).__init__(None, None)
        if isinstance(iterator1, iterator_module.Iterator):
            iterator = {'main': iterator1, 'second': iterator2}  # イテレータの登録
            self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}  # オプティマイザの設定
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)
        self.model = Mymodel
        self.converter = converter  # GPUに転送する用意
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0
        self.accfun = accuracy.accuracy
        self.loss = None
        self.accuracy = None

    def update_core(self):
        model = self.model
        batch1 = self._iterators['main'].next()  # 同じデータを並列して流す
        batch2 = self._iterators['second'].next()

        in_arrays1 = self.converter(batch1, self.device)
        in_arrays2 = self.converter(batch2, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target


        if isinstance(in_arrays1, tuple):
            # optimizer.update(loss_func, *in_arrays1)
            # optimizer.update(loss_func, *in_arrays2)

            # mini_batch_list = [in_arrays1[0], in_arrays2[0], in_arrays1[1]]
            # optimizer.update(loss_func, mini_batch_list)
            # mini_batch_list = []

            y = model(in_arrays1[0], in_arrays2[0])
            loss = F.softmax_cross_entropy(y, in_arrays1[1])
            # print("loss = " + str(loss.data))
            #   print ("re")
            acc = self.accfun(y, in_arrays1[1])
            reporter.report({'accuracy': acc, 'loss': loss})

            # calc = int(self.epoch) % 20

            # if self.epoch == 20:
            #      print('epoch:{:02d}   main/accuracy:{:.09}   main/loss:{:.09}'.format(
            #         self.epoch, accuracy.data, loss.data))

            model.cleargrads()
            loss.backward()
            optimizer.update()
            # print(str(count))

        elif isinstance(in_arrays1, dict):
            optimizer.update(loss_func, **in_arrays1)
            optimizer.update(loss_func, **in_arrays2)
        else:
            optimizer.update(loss_func, in_arrays1)
            optimizer.update(loss_func, in_arrays2)

class MyClasifier(chainer.Chain):
    def __init__(self, model):
        super(MyClasifier, self).__init__(predictor=model)
        self.predictor = model
        self.accfun = accuracy.accuracy
        self.accuracy = None
        self.loss = None

    def __call__(self, mini_batch_list):

        data1 = mini_batch_list[0]
        data2 = mini_batch_list[1]
        labels = mini_batch_list[2]
        y = self.predictor(data1, data2)
        self.loss = F.SoftmaxCrossEntropy(y, labels)
        reporter.report({'main/loss': self.loss}, self)

        self.accuracy = self.accfun(y, labels)
        reporter.report({'main/accuracy': self.accuracy}, self)
        return self.loss





# train model
def train():

    Dt1_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd"

    train1 = load_image(path_dir=Dt1_train_dir, mode=True)

    Dt2_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt2_(Gallery&Probe)"
    train2 = load_image(path_dir=Dt2_train_dir, mode=True)

    model = Multi_modal_GEINet()

    model.to_gpu()


    Dt1_train_iter = iterators.SerialIterator(train1, batch_size=239, shuffle=False)
    Dt2_train_iter = iterators.SerialIterator(train2, batch_size=239, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.02, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

    # updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'second': 1})
    updater = Multi_modal_Updater(model, Dt1_train_iter, Dt2_train_iter, optimizer, device=0)
    epoch = 6250

    trainer = training.Trainer(updater, (epoch, 'epoch'),
                               out='/home/wutong/Setoguchi/chainer_files/result')

    # trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.ExponentialShift(attr='lr', rate=0.56234), trigger=(1250, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='SFDEI_log', trigger=(50, "epoch")))
    trainer.extend(extensions.snapshot(), trigger=(1250, 'epoch'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_{.updater.epoch}'), trigger=(1250, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch',
                                           'accuracy',
                                           'loss']))
    # 'validation/main/accuracy']),
    # trigger=(1, "epoch"))
    trainer.extend(extensions.dump_graph(root_name="loss", out_name="multi_modal.dot"))
    trainer.extend(extensions.PlotReport(["loss"]), trigger=(20, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # Run the trainer
    trainer.run()