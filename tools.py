# _*_ coding:utf-8 _*_
import os
import shutil
import csv
import chainer
from chainer import dataset
import glob
import cupy as cp
import numpy as np
caffe_root = '/home/common-ns/caffe-master/'
import sys
# caffeのpythonモジュールのパスを設定
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
from itertools import chain


# いくつかの

# ファイルをディレクトリ毎コピーする
# GalleryとProbeからCV01とCV02を分ける

def dir_copy():

    cv01_name = open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV01.txt", mode='r')
    cv02_name = open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV02.txt", mode='r')

    cv01 = cv01_name.readlines()
    cv02 = cv02_name.readlines()

    for list1 in cv01:
         shutil.copytree("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/ALL/" + list1.replace("\n", ""),
                        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01/" + list1.replace("\n", ""))

    for list2 in cv02:
         shutil.copytree("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/ALL/" + list2.replace("\n", ""),
                        "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02/" + list2.replace("\n", ""))

    print("complete copy")
# -----------------------------------------------------------------------------------------------------

# ディレクトリの名前を変更

def change_dir_name():
    cv01_name = open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV02.txt", mode='r')
    cv01 = cv01_name.readlines()

    for list1 in cv01:
        tmp = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02/" + list1
        os.rename(tmp, tmp.replace("\n", ""))
# ---------------------------------------------------------------------------


# OULPから各オブジェクトの１歩行周期分を抜き出してコピー

def copy_gait_cycle():

    f1 = open("/home/wutong/Files/OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-55_Gallery.csv"
             , 'r')
    fread1 = csv.reader(f1)

    for line in fread1:

        os.mkdir("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/ALL/" + line[0])
        for item in range(int(line[2]), int(line[3])+1):
            num = "{0:08d}".format(item)
            shutil.copyfile("/home/wutong/Files/OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(88x128)/Seq00/" + line[0] +
                            "/" + num + ".png" ,
                            "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/ALL/" + line[0] + "/"
                            + num + ".png")
    print("finished Gallery")

    f2 = open("/home/wutong/Files/OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-55_Probe.csv"
             , 'r')
    fread2 = csv.reader(f2)

    for line in fread2:
        os.mkdir("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/ALL/" + line[0])
        for item in range(int(line[2]), int(line[3])+1):
            num = "{0:08d}".format(item)
            shutil.copyfile("/home/wutong/Files/OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(88x128)/Seq01/" + line[0] +
                            "/" + num + ".png" ,
                            "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/ALL/" + line[0] + "/"
                            + num + ".png")

    print("finished Probe")
# -----------------------------------------------------------------------------------------------------


# textfileに画像データのパスとラベルを書き込む

def write_file(path_txt, path_dir):
    file_name = open(path_txt, 'r')
    inline = file_name.readlines()
    label = 0
    count = 0

    for line in inline:
        line = line.replace("\n", "")
        itemname = path_dir + line +"/*png"
        all_item = sorted(glob.glob(itemname))
        # sum += len(all_item)
        # print("全ての画像は" + str(sum) + "枚です")

        # imagedata = []
        # labeldata =[]

        for item in all_item:
            if (label == 0) and (count == 0):
                with open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01_gait_cycle_all.txt", "w") \
                     as f:
                     f.write(item + " " + str(label) + "\n")
                # with open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02_gait_cycle_all_label.txt", "w") \
                #     as f:
                #     f.write(str(label) + "\n")
                count += 1

            else:
                with open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01_gait_cycle_all.txt", "a") \
                     as f:
                     f.write(item + " " + str(label) + "\n")

                # with open("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02_gait_cycle_all_label.txt", "a") \
                #     as f:
                #     f.write(str(label) + "\n")

        label += 1
# -------------------------------------------------------------------------------------------


# 異なるフレーム間隔のsfdeiを一つにくっつける
def conect_sfdei(type):
    dt1 = "/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/" + type  +"/signed/128_3ch/CV02_(" + type + ")/128/Dt1"


    filename = glob.glob(dt1 + "/*")
    for item in filename:
        dt1 = np.load(item)
        dt2 = np.load(item.replace("Dt1", "Dt2"))
        dt3 = np.load(item.replace("Dt1", "Dt3"))
        dt4 = np.load(item.replace("Dt1", "Dt4"))
        dt5 = np.load(item.replace("Dt1", "Dt5"))
        dt6 = np.load(item.replace("Dt1", "Dt6"))
        dt7 = np.load(item.replace("Dt1", "Dt7"))
        dt8 = np.load(item.replace("Dt1", "Dt8"))
        dt9 = np.load(item.replace("Dt1", "Dt9"))
        dt10 = np.load(item.replace("Dt1", "Dt10"))

        new = np.dstack((dt1, dt2))
        new = np.dstack((new, dt3))
        new = np.dstack((new, dt4))
        new = np.dstack((new, dt5))
        new = np.dstack((new, dt6))
        new = np.dstack((new, dt7))
        new = np.dstack((new, dt8))
        new = np.dstack((new, dt9))
        new = np.dstack((new, dt10))
        np.save("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/" + type + "/signed/128_3ch/CV02_conect_AllDt(" + type + ")/" +
                item.replace("/media/wutong/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/" + type  +"/signed/128_3ch/CV02_(" + type + ")/128/Dt1/", ""), new)
        # print(new.shape)
# -----------------------------------------------------------------------------------------------


# lmdbファイルをつくる
def make_lmdb():

    data_path = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd"
    all_data = glob.glob(data_path + "/*")
    N = len(all_data)
    x = np.zeros((N, 3, 128, 88), dtype=np.float32)  # データ用の変数を用意
    y = np.zeros(N, dtype=np.int64)  # ラベル用の変数を用意
    label = 0

    # 用意した変数にロードしたデータとラベルを格納
    for index, item in enumerate(all_data):
        data = np.load(item)
        data = data.transpose([2, 0, 1])  # 元データは(縦×横×チャンネル)なので、(チャンネル×縦×横)に変える
        x[index] = data
        if index % 2 == 0:
            label += 1
        print(str(label-1))
        y[index] = label-1

    print("x len = " + str(len(x)))
    print("y len = " + str(len(y)))
    print(str(np.max(x[48])))
    print(str(np.max(y[48])))

    map_size = x.nbytes * 10  # numpyデータのトータルサイズの十倍のサイズを確保

    env = lmdb.open(
        '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/LMDB/otamesi',
        map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        cnt = 0
        for i in range(N):
            # print(str(i))
            datum = caffe.proto.caffe_pb2.Datum()  # 構造体生成
            # print(x.shape)
            datum.channels = 3  # x.shape[1]
            datum.height = 128  # x.shape[2]
            datum.width = 88  # x.shape[3]
            datum.data = x[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            # str_id = '{:08}'.format(i)
            str_id = '{:0>08d}'.format(i)

            # txn.put(str_id, datum.SerializeToString())
            # txn.put(datum.SerializeToString())
            
            # The encode is only essential in Python 3
            # txn.put(str_id.encode('ascii'), datum.SerializeToString())
            txn.put(str_id, datum.SerializeToString())  # Datumインスタンス全体をバイト列にしつつデータベースに送信
    print "a"
    # '''


# ---------------------------------------------------------------------------------------------

# lmdbを開く
def open_lmdb():
    list = []
    raw_datum = []
    env = lmdb.open('/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/LMDB/otamesi', readonly=True)

    with env.begin() as txn:
        # raw_dataum = txn.get(b'00000000')
        for i in range(1912) :
            raw_datum = txn.get(b'0000' + str(i))

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)

            flat_x = np.fromstring(datum.data, dtype=np.float32)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            list.append(x, y)

    print("a")
# --------------------------------------------------------------------------------

# GEI or SFDEIのための学習/テストデータの作成とロード
def load_GEI(path_dir, mode):
    # -----------------------------------------------------------------------------
    if mode:  # 学習用データセットの準備
        fnames = glob.glob(
            "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/signed/128_3ch/CV01_(Gallery&Probe)_2nd/*")
        #  file = [(cp.load(f)/255.0).astype(cp.float32).transpose([2, 0, 1]) for f in fnames]
        file = []
        for index, item in enumerate(fnames):
            item = np.load(item).astype(np.float32).transpose([2, 0, 1]) / 255.0
            item = item[(2, 1, 0), :, :]
            # item = item = item.reshape((1, item.shape[0], item.shape[1], item.shape[2]))
            file.append(item)
        # label = [os.path.basename(f) for f in fnames]
        labels = []
        for num in range(len(fnames) / 2):
            labels.append(np.int32(num))
            labels.append(np.int32(num))

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
            # item = item[(2, 1, 0), :, :]

            item = item.reshape((1, item.shape[0], item.shape[1], item.shape[2]))
            data.append(item)
            print (int(index))
            labels.append(int(index))
        # data = np.array(data)
        # data = chainer.Variable(data)
        # print(labels)
        return data, labels

# -------------------------------------------------------------------------------------------
# OU-ISIRの内の歩容データのロード、学習/テストデータの作成
# フォルダごとに人物IDが異なるので、同じフォルダ内の画像全てに同じラベルが付与されるように処理

def load_OULP(path_dir):
    # 各オブジェクトごとのフォルダ
    dnames = sorted(glob.glob(path_dir))

    # 画像ファイルパス一覧
    fnames = [sorted(glob.glob('{}/*.png'.format(d))) for d in dnames]
    #  if not os.path.exists('{}/ignore'.format(d))]
    fnames = list(chain.from_iterable(fnames))

    # それぞれにフォルダ名から一意なIDを付与
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    dnames = [os.path.basename(d) for d in dnames]
    # if not os.path.exists('{}/ignore'.format(d))]
    labels = [dnames.index(l) for l in labels]
    # データセット作成
    d = chainer.datasets.LabeledImageDataset(list(zip(fnames, labels)))

    return d  # 画像のパスとラベルの組のリストを返す
