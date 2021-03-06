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

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# imagefileをコピーして、ラベルのタプルにする
def load_image(path_dir, mode):
    '''
    # -----------------------------------------------------------------------------
    file_name = open(path_txt, 'r')
    inline = file_name.readlines()
    label = 0
    datasets = []
    # sum = 0
    for line in inline:
        line = line.replace("\n", "")
        itemname = path_dir + line +"/*png"
        all_item = glob.glob(itemname)
        # sum += len(all_item)
        # print("全ての画像は" + str(sum) + "枚です")
        count = 0
        imagedata = []
        labeldata =[]

        for item in all_item:
            img = cv2.imread(item, flags=cv2.IMREAD_GRAYSCALE)
            # resize_img = cv2.resize(img, (88, 128))
            x = np.array(img, dtype=np.float32)
            tmp = x.reshape(1, 128, 88)     # (チャンネル数, 高さ, 幅)
            tmp = tmp/255.0
            imagedata.append(tmp)

            t = np.array(label, dtype=np.int32)
            labeldata.append(t)
            # print(item)
            datasets.append((tmp, t))
            # print(datasets)


        label += 1

    print(label)
    if label == 956:  # 0~955が実際にラベリングされる
        print(len(datasets))
        print("load completed")
    return datasets
    # ----------------------------------------------------------------------------------


    # 各オブジェクトごとのフォルダ
    dnames = glob.glob("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01/*")

    # 画像ファイルパス一覧
    fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames]
                #  if not os.path.exists('{}/ignore'.format(d))]
    fnames = list(chain.from_iterable(fnames))

    # それぞれにフォルダ名から一意なIDを付与
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    dnames = [os.path.basename(d) for d in dnames]
               # if not os.path.exists('{}/ignore'.format(d))]
    labels = [dnames.index(l) for l in labels]
    # データセット作成
    d = chainer.datasets.LabeledImageDataset(list(zip(fnames, labels)))
    return d
    '''
    # -----------------------------------------------------------------------------
    if mode:  # 学習用データセットの準備
        fnames = sorted(glob.glob(path_dir + "/*"))
        #  file = [(cp.load(f)/255.0).astype(cp.float32).transpose([2, 0, 1]) for f in fnames]
        file = []
        for index, item in enumerate(fnames):
            item = cp.load(item).astype(cp.float32).transpose([2, 0, 1]) / 255.0
            file.append(item)
        # label = [os.path.basename(f) for f in fnames]
        labels = []
        for num in range(len(fnames) / 2):
            labels.append(cp.int32(num))
            labels.append(cp.int32(num))

        # List = list(zip(file, labels))
        # data = chainer.datasets.TupleDataset(List)
        data = chainer.datasets.TupleDataset(file, labels)
        return data
        # -------------------------------------------------------------------------------
    if not mode:  # 識別用データセットの準備
        data = []
        labels = []
        filelist = sorted(glob.glob(path_dir + "/*"))
        for index, item in enumerate(filelist):
            item = np.load(item).astype(np.float32).transpose([2, 0, 1]) / 255.0
            item = item.reshape((1, item.shape[0], item.shape[1], item.shape[2]))
            data.append(item)
            labels.append(int(index))

        return data, labels


class GEINet(Chain):
    def __init__(self):
        super(GEINet, self).__init__()
        self.W = initializers.HeNormal()
        self.b = initializers.Constant(fill_value=0)
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=2, in_channels=3, out_channels=18, ksize=7, stride=1, initialW=self.W, initial_bias=self.b)
            self.conv2 = L.ConvolutionND(ndim=2, in_channels=18, out_channels=45, ksize=5, stride=1, pad=(2, 2), initialW=self.W, initial_bias=self.b)
            self.fc3 = L.Linear(None, 1024, initialW=self.W, initial_bias=self.b)
            self.fc4 = L.Linear(None, 956, initialW=self.W, initial_bias=self.b)
            # L.Deconvolution2D()

    def __call__(self, x, y):
        root1_h1 = F.relu(self.conv1(x))
        root1_h2 = F.max_pooling_2d(root1_h1, stride=2, ksize=2)
        root1_h3 = F.local_response_normalization(root1_h2, k=1, n=5, alpha=0.00002, beta=0.75)
        root1_h4 = F.relu(self.conv2(root1_h3))
        root1_h5 = F.max_pooling_2d(root1_h4, stride=2, ksize=3)
        root1_h6 = F.local_response_normalization(root1_h5, k=1, n=5, alpha=0.00002, beta=0.75)

        root2_h1 = F.relu(self.conv1(y))
        root2_h2 = F.max_pooling_2d(root2_h1, stride=2, ksize=2)
        root2_h3 = F.local_response_normalization(root2_h2, k=1, n=5, alpha=0.00002, beta=0.75)
        root2_h4 = F.relu(self.conv2(root2_h3))
        root2_h5 = F.max_pooling_2d(root2_h4, stride=2, ksize=3)
        root2_h6 = F.local_response_normalization(root2_h5, k=1, n=5, alpha=0.00002, beta=0.75)

        h6 = F.concat((root1_h6, root2_h6), axis=1)

        h7 = F.dropout(F.relu(self.fc3(h6)), ratio=0.5)
        h8 = self.fc4(h7)
        # h9 = F.softmax(h8)
        return h8

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
    # train_txt = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV01.txt"
    # train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd"
    Dt1_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd"
    # Dt2_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt2(Gallery&Probe)"


    # train = load_image(path_dir=train_dir, mode=True)
    train1 = load_image(path_dir=Dt1_train_dir, mode=True)

    Dt2_train_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_Dt2_(Gallery&Probe)"
    train2 = load_image(path_dir=Dt2_train_dir, mode=True)

    # print(train[0])

    # test_txt = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV02.txt"
    # test_dir = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02/"
    # test = load_image(path_dir=test_dir, mode=True)

    # print(test[0])

    # quit()
    # model = MyClasifier(GEINet())
    model = GEINet()

    model.to_gpu()

    # train_iter = iterators.MultiprocessIterator(train, batch_size=239)
    Dt1_train_iter = iterators.SerialIterator(train1, batch_size=239, shuffle=False)
    Dt2_train_iter = iterators.SerialIterator(train2, batch_size=239, shuffle=False)

    # from IPython import embed
    # embed()
    # test_iter = iterators.MultiprocessIterator(test, batch_size=239, repeat=False, shuffle=False)

    # optimizer = chainer.optimizers.SGD(lr=0.02)
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
    trainer.extend(extensions.PlotReport(["loss"]), trigger=(50, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # Run the trainer
    trainer.run()


    '''
    logs = []

    max_epoch = 50000

    while Dt1_train_iter.epoch < max_epoch:

        # ---------- One iteration of the training loop ----------
        root1_batch = Dt1_train_iter.next()
        root2_batch = Dt2_train_iter.next()
        print(root1_batch[1])
        print(root2_batch[1])
        quit()

        Dt1, target_train = chainer.dataset.concat_examples(root1_batch, 0)
        Dt2, target_train = chainer.dataset.concat_examples(root2_batch, 0)

        # Calculate the prediction of the network
        prediction_train = model(Dt1, Dt2)

        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, target_train)

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # Update all the trainable paremters
        optimizer.update()
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if Dt1_train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04} '.format(
                Dt1_train_iter.epoch, to_cpu(loss.data)))
    '''
    serializers.save_npz("/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model", model)
    '''
    df = pd.DataFrame(logs)
    df.to_csv("/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/loss.csv")
    '''

def extract_features(model, data1, data2):
    feature_list = np.empty((0, 1024), dtype=float)
    # count = 0
    for i in range(len(data1)):
        # Forward処理を記述
        root1_h1 = F.relu(model.conv1(data1[i]))
        root1_h2 = F.max_pooling_2d(root1_h1, stride=2, ksize=2)
        root1_h3 = F.local_response_normalization(root1_h2, n=5, alpha=0.0001, beta=0.75)
        root1_h4 = F.relu(model.conv2(root1_h3))
        root1_h5 = F.max_pooling_2d(root1_h4, stride=2, ksize=3)
        root1_h6 = F.local_response_normalization(root1_h5, n=5, alpha=0.0001, beta=0.75)

        root2_h1 = F.relu(model.conv1(data2[i]))
        root2_h2 = F.max_pooling_2d(root2_h1, stride=2, ksize=2)
        root2_h3 = F.local_response_normalization(root2_h2, n=5, alpha=0.0001, beta=0.75)
        root2_h4 = F.relu(model.conv2(root2_h3))
        root2_h5 = F.max_pooling_2d(root2_h4, stride=2, ksize=3)
        root2_h6 = F.local_response_normalization(root2_h5, n=5, alpha=0.0001, beta=0.75)

        h6 = F.concat((root1_h6, root2_h6), axis=1)

        feature = model.fc3(h6)
        # h8 = model.fc4(h7)

        print("fc3層の出力は：" + str(feature.data[0]))
        feature_list = np.append(feature_list, np.array([feature.data[0]]), axis=0)
    # feature_list = feature_list.flatten


    return feature_list


def recognition(model_name):
    # 識別用のデータをダウンロード
    train1, train_labels1 = load_image(
        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV02(Gallery)_2nd",
        mode=False)
    train2, train_labels2 = load_image(
        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV02_Dt2(Gallery)",
        mode=False)
    test1, test_labels1 = load_image(
        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/signed/128_3ch/CV02(Probe)_2nd",
        mode=False)
    test2, test_labels2 = load_image(
        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/signed/128_3ch/CV02_Dt2(Probe)",
        mode=False)



    # extract features
    model = GEINet()
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



# train()
recognition('/home/wutong/Setoguchi/chainer_files/SFDEINet_multi_modal/SFDEINet_multi_modal_model_90000')
