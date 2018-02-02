# coding:utf-8

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
sys.path.append('/home/common/setoguchi/chainer_files/SFDEINet_multi_modal/Multi_modal_GEINet-master')
from tools import load_GEI
sys.path.append('/home/common/setoguchi/chainer_files/SFDEINet_multi_modal/Multi_modal_GEINet-master')
from Multi_modal_GEINet import Multi_modal_GEINet
from sklearn.neighbors import KNeighborsClassifier

def extract_features(model, data1, data2):
        root1_h1 = F.relu(self.conv1(x))
        root1_h2 = F.max_pooling_2d(root1_h1, stride=2, ksize=2)
        root1_h3 = F.local_response_normalization(root1_h2, k=1, n=5, alpha=0.00002, beta=0.75)
        root1_h4 = F.relu(self.conv2(root1_h3))
        root1_h5 = F.max_pooling_2d(root1_h4, stride=2, ksize=3)
        root1_h6 = F.local_response_normalization(root1_h5, k=1, n=5, alpha=0.00002, beta=0.75)
        root1_h7 = F.relu(self.fc3(root1_h6))
        
        root2_h1 = F.relu(self.conv1(y))
        root2_h2 = F.max_pooling_2d(root2_h1, stride=2, ksize=2)
        root2_h3 = F.local_response_normalization(root2_h2, k=1, n=5, alpha=0.00002, beta=0.75)
        root2_h4 = F.relu(self.conv2(root2_h3))
        root2_h5 = F.max_pooling_2d(root2_h4, stride=2, ksize=3)
        root2_h6 = F.local_response_normalization(root2_h5, k=1, n=5, alpha=0.00002, beta=0.75)
        root2_h7 = F.relu(self.fc3(root2_h6))

        feature = F.concat((root1_h7, root2_h7), axis=1)
        
        h8 = self.fc4(h7)
        # h9 = F.softmax(h8)

        # print("fc3層の出力は：" + str(feature.data[0]))
        feature_list = np.append(feature_list, np.array([feature.data[0]]), axis=0)
    # feature_list = feature_list.flatten


    return feature_list


def recognition(model_name):
    # 識別用のデータをダウンロード
    train1, train_labels1 = load_GEI(
        "/home/common/setoguchi/chainer_files/dataset/SFDEI/CV02(Gallery)_2nd",
        mode=False)
    train2, train_labels2 = load_GEI(
        "/home/common/setoguchi/chainer_files/dataset/SFDEI/CV02_Dt2(Gallery)",
        mode=False)
    test1, test_labels1 = load_GEI(
        "/home/common/setoguchi/chainer_files/dataset/SFDEI/CV02(Probe)_2nd",
        mode=False)
    test2, test_labels2 = load_GEI(
        "/home/common/setoguchi/chainer_files/dataset/SFDEI/CV02_Dt2(Probe)",
        mode=False)



    # extract features
    model = Multi_modal_GEINet()
    serializers.load_npz(model_name, obj=model)

    train_features = extract_features(model, train1, train2)
    test_features = extract_features(model, test1, test2)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_features, train_labels1)

    correct = 0.0
    for i, item in enumerate(test_features):
        predit = neigh.predict([item])[0]
        print "label:%d, predict:%d" % (test_labels1[i], predit)
        if predit == test_labels1[i]:
            correct = correct + 1
    # else:
    #         f.write("label:%d, predict:%d" %(true_label[i],predit)+"\n")

    acc = correct / len(test_features)
    print acc

if __name__=='__main__':
    recognition('/home/common/setoguchi/chainer_files/SFDEINet_multi_modal/Multi_modal_GEINet-master/result_late/model_snapshot_6250')
