"""
    @author:tb
    @time:2024-5-10
    模型训练与评估
    更新：多种评估方法并绘图
"""
import json
import os
import time

import numpy as np

from core.model import Model
from core.plt_picture import PicturePlt
from core.data_processor import inverse_normalise_window
from keras.api.utils import plot_model
from core.data_processor import DataLoader
from sklearn.preprocessing import MinMaxScaler


def main(train=True):
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.mkdir(configs["model"]["save_dir"])
    # 数据读取
    data = DataLoader(
        os.path.join("data", configs["data"]["filename"]),
        configs["data"]["train_test_split"],
        configs["data"]["columns"]
    )

    # 加载训练数据
    x, y = data.get_train_data(
        seq_len=configs["data"]["sequence_length"],
        normalise=configs["data"]["normalise"]
    )
    # 加载测试数据
    x_test, y_test = data.get_test_data(
        seq_len=configs["data"]["sequence_length"],
        normalise=configs["data"]["normalise"]
    )
    train_loss_list = []
    val_loss_list = []
    path = input("输入实验次数：")
    exp = f"result/exp{path}"
    loss_path = f"{exp}/{configs['model']['loss']}"
    # RNN模型创建与模型图绘制
    model = Model()
    if train:
        rnn_model = model.build_model(configs)
        plot_model(rnn_model, to_file="result/model.jpg", show_shapes=True)

        # 模型训练
        train_list, val_list = model.train(
            x,
            y,
            x_test,
            y_test,
            epochs=configs["training"]["epochs"],
            batch_size=configs["training"]["batch_size"],
            save_dir=configs["model"]["save_dir"],
            optimizer=configs["model"]["optimizer"]
        )
        train_loss_list.extend(train_list)
        val_loss_list.extend(val_list)
        losses_path = f"loss/exp{path}"
        if not os.path.exists(losses_path):
            os.mkdir(losses_path)
        train_loss = f"exp{path}/{configs['model']['optimizer']}-{configs['model']['loss']}-train_loss"
        val_loss = f"exp{path}/{configs['model']['optimizer']}-{configs['model']['loss']}-val_loss"
        data.save_loss(train_list, train_loss)
        data.save_loss(val_list, val_loss)
    else:
        model_name = input("输入想要加载的模型名字：")
        path = f"saved_models/{model_name}"
        model = model.load_model(path)
        if model is not None:
            print("[Load Model Success !]")
        else:
            print('模型加载错误')
        # 模型评估
        print("[Model Evaluate] start -----")
        # model.model_evaluate(x_test, y_test, name=configs["model"]["loss"])

    print("[Plot Picture] start -----")
    time.sleep(1)

    # 可视化
    plt_img = PicturePlt()

    # 这里是序列->序列的预测图，需要就取消注释
    # predictions_multiple = model.predict_sequences_multiple(x_test, configs["data"]["sequence_length"],
    #                                                         configs["data"]["sequence_length"])
    # print(np.array(predictions_multiple).shape)
    # plt_img.plot_results_multiple(predictions_multiple, y_test, configs["data"]["sequence_length"])

    # 序列 -> 点的预测图
    predictions_single = model.predict_point_by_point(x_test)
    plt_img.plot_results(predictions_single, y_test, name=f'{configs["model"]["optimizer"]}-{configs["model"]["loss"]}',
                         path=exp)
    # 本次实验训练集和测试集损失率的图
    plt_img.plot_loss_single(train_loss_list, val_loss_list, loss_path)

    # 绘制多张图,需要csv大于4
    # 传参single=True就是在n个损失函数，False就是n个优化器模型，默认True
    # plt_img.plot_loss_multiple(exp)


if __name__ == '__main__':
    main(True)
