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
from chainer.backends import cuda
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import sys
sys.path.append('/home/common-ns/PycharmProjects/Prepare')
from tools import load_GEI
sys.path.append('/home/common-ns/PycharmProjects/models')
from Multi_modal_GEINet_with_9branch import Multi_modal_GEINet



class Multi_modal_Updater(training.StandardUpdater):
    def __init__(self, Mymodel, iterator1, iterator2, iterator3, iterator4, iterator5,
                 iterator6, iterator7, iterator8, iterator9, optimizer, converter=convert.concat_examples,
                 device=None,
                 loss_func=None):
        super(Multi_modal_Updater, self).__init__(None, None)
        if isinstance(iterator1, iterator_module.Iterator):
            iterator = {'main': iterator1, 'second': iterator2, 'third': iterator3, '4th': iterator4,
                        '5th': iterator5, '6th': iterator6, '7th': iterator7, '8th': iterator8,
                        '9th': iterator9}
            self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)
        self.model = Mymodel
        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0
        self.accfun = accuracy.accuracy
        self.loss = None
        self.accuracy = None

    def update_core(self):
        model = self.model
        batch1 = self._iterators['main'].next()
        batch2 = self._iterators['second'].next()
        batch3 = self._iterators['third'].next()
        batch4 = self._iterators['4th'].next()
        batch5 = self._iterators['5th'].next()
        batch6 = self._iterators['6th'].next()
        batch7 = self._iterators['7th'].next()
        batch8 = self._iterators['8th'].next()
        batch9 = self._iterators['9th'].next()

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target


        sum_acc = 0
        accum_loss = 0
        # 新しい勾配計算に入る前に、以前の蓄積勾配をクリアにしておく
        model.cleargrads()
        for i in range(0, len(batch1)):
        # dataだけ取り出してgpuで計算
            data1 = chainer.Variable(cuda.to_gpu(batch1[i][0].reshape((1, 3, 128, 88))))
            data2 = chainer.Variable(cuda.to_gpu(batch2[i][0].reshape((1, 3, 128, 88))))
            data3 = chainer.Variable(cuda.to_gpu(batch3[i][0].reshape((1, 3, 128, 88))))
            data4 = chainer.Variable(cuda.to_gpu(batch4[i][0].reshape((1, 3, 128, 88))))
            data5 = chainer.Variable(cuda.to_gpu(batch5[i][0].reshape((1, 3, 128, 88))))
            data6 = chainer.Variable(cuda.to_gpu(batch6[i][0].reshape((1, 3, 128, 88))))
            data7 = chainer.Variable(cuda.to_gpu(batch7[i][0].reshape((1, 3, 128, 88))))
            data8 = chainer.Variable(cuda.to_gpu(batch8[i][0].reshape((1, 3, 128, 88))))
            data9 = chainer.Variable(cuda.to_gpu(batch9[i][0].reshape((1, 3, 128, 88))))

            # ラベルのみ取り出す。chainerのdataのbatch数(0次元目)とラベルのshapeを擬似的に統一
            # この処理を加えないとtype_check関数で怒られる
            label = np.array(batch1[i][1]).reshape((1,))
            label = cuda.to_gpu(label)
            output = model(data1, data2, data3, data4, data5, data6, data7, data8, data9)
            loss = F.softmax_cross_entropy(output, label)
            accum_loss += loss
            sum_acc += self.accfun(output, label)
            loss.backward()

        reporter.report({'accuracy': sum_acc / len(batch1[0]), 'loss': accum_loss})

        optimizer.update()


# ここでは使ってない
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
def train(mode):

    Dt1_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd"
    train1 = load_GEI(path_dir=Dt1_train_dir, mode=True)

    Dt2_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt2_(Gallery&Probe)"
    train2 = load_GEI(path_dir=Dt2_train_dir, mode=True)

    Dt3_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt3_(Gallery&Probe)"
    train3 = load_GEI(path_dir=Dt3_train_dir, mode=True)

    Dt4_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt4_(Gallery&Probe)"
    train4 = load_GEI(path_dir=Dt4_train_dir, mode=True)

    Dt5_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt5_(Gallery&Probe)"
    train5 = load_GEI(path_dir=Dt5_train_dir, mode=True)

    Dt6_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt6_(Gallery&Probe)"
    train6 = load_GEI(path_dir=Dt6_train_dir, mode=True)

    Dt7_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt7_(Gallery&Probe)"
    train7 = load_GEI(path_dir=Dt7_train_dir, mode=True)

    Dt8_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt8_(Gallery&Probe)"
    train8 = load_GEI(path_dir=Dt8_train_dir, mode=True)

    Dt9_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt9_(Gallery&Probe)"
    train9 = load_GEI(path_dir=Dt9_train_dir, mode=True)

    model = Multi_modal_GEINet()

    model.to_gpu()

    # train_iter = iterators.MultiprocessIterator(train, batch_size=239)
    Dt1_train_iter = iterators.SerialIterator(train1, batch_size=239, shuffle=False)
    Dt2_train_iter = iterators.SerialIterator(train2, batch_size=239, shuffle=False)
    Dt3_train_iter = iterators.SerialIterator(train3, batch_size=239, shuffle=False)
    Dt4_train_iter = iterators.SerialIterator(train4, batch_size=239, shuffle=False)
    Dt5_train_iter = iterators.SerialIterator(train5, batch_size=239, shuffle=False)
    Dt6_train_iter = iterators.SerialIterator(train6, batch_size=239, shuffle=False)
    Dt7_train_iter = iterators.SerialIterator(train7, batch_size=239, shuffle=False)
    Dt8_train_iter = iterators.SerialIterator(train8, batch_size=239, shuffle=False)
    Dt9_train_iter = iterators.SerialIterator(train9, batch_size=239, shuffle=False)


    # optimizer = chainer.optimizers.SGD(lr=0.02)
    optimizer = chainer.optimizers.MomentumSGD(lr=0.02, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

    # updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'second': 1})
    updater = Multi_modal_Updater(model, Dt1_train_iter, Dt2_train_iter, Dt3_train_iter,
                                  Dt4_train_iter, Dt5_train_iter, Dt6_train_iter, Dt7_train_iter,
                                  Dt8_train_iter, Dt9_train_iter, optimizer, device=0)
    epoch = 6250

    trainer = training.Trainer(updater, (epoch, 'epoch'),
                               out='/home/wutong/Setoguchi/chainer_files/result')

    # trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.ExponentialShift(attr='lr', rate=0.56234), trigger=(1250, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='SFDEI_log', trigger=(20, "epoch")))
    trainer.extend((extensions.snapshot_object(model, filename='model_shapshot_{.update.epoch}')), trigger=(1250, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(1250, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch',
                                           'accuracy',
                                           'loss']))
    # 'validation/main/accuracy']),
    # trigger=(1, "epoch"))
    trainer.extend(extensions.dump_graph(root_name="loss", out_name="multi_modal_3.dot"))
    trainer.extend(extensions.PlotReport(["loss"]), trigger=(50, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    if mode ==True:
        # Run the trainer
        trainer.run()
    else:
        serializers.load_npz("/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model",
                             trainer)
        trainer.run()
        serializers.save_npz("/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model",
                             trainer)

    serializers.save_npz("/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model", model)


if __name__=='__main__':
    train(mode=True)
    # recognition('/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model_40000')
