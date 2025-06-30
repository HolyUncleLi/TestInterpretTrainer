import numpy.linalg
import torch.linalg
import sklearn
import gc
import shap
import glob
from InterpretTrainer_Base import *
from AttnBaseModel_InterpretTrain import MainModel, FeatureProcess, Head

import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

EPOCH = 200

BATCH_SIZE = 64
seq_len = 1

LEARNING_RATE = 0.0005
use_gpu = torch.cuda.is_available()

folder = "./SleepEdfData/SCDataSet/data/"
# h5file = "./SleepEdfData/SCDataset/data/"
model_path = ""
# files = os.listdir(h5file)
# files_len = len(files)
keys = ["Fpz-Cz", "Pz-Oz", "label"]
band_name = ['delta', 'theta', 'alpha', 'sigma', 'beta']
'''
modelPath = './result/module/'
model_files = os.listdir(modelPath)
model_files_len = len(model_files)
'''

kfold = 20


if __name__ == '__main__':

    index = 0

    kf = sklearn.model_selection.GroupKFold(n_splits=kfold)
    # eeg_data, labels, groups = getEEGData_group(h5file, files, channel=0)
    eeg_data, labels, groups = load_and_concat_npz(folder)
    # print("groups : ", len(groups), groups)

    count_preClass = checkDataset(labels)
    classWeight = [1,1.5,1,1,1]

    print(">>>")
    print('data  size: ',eeg_data.shape)
    print('label size: ',labels.shape)
    print('count pre class: ', count_preClass)
    print("weight pre class: ", classWeight)
    print('group: ', groups)
    print('>>>')

    # for train_index, test_index in kf.split(eeg_data):
    for train_index, test_index in kf.split(eeg_data, groups=groups):

        torch.cuda.empty_cache()
        if index >= 0:
            # 分别打乱数据
            x_train, x_test = eeg_data[train_index], eeg_data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            x_train, y_train = cutData(x_train, y_train, size=seq_len)
            x_train = x_train[0:1500].reshape(-1, seq_len, 3000)
            y_train = y_train[0:1500].reshape(-1, seq_len, 1)
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

            x_test, y_test = cutData(x_test[0:250], y_test[0:250], size=seq_len)
            x_test = x_test.reshape(-1, seq_len, 3000)
            y_test = y_test.reshape(-1, seq_len, 1)

            print(x_train.shape, y_train.shape)
            print(x_test.shape, y_test.shape)

            # 模型1---处理数据
            model = MainModel()

            # 模型2---分类
            ftcnn = FeatureProcess().cuda()
            classhead = Head()

            if use_gpu:
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                loss_func = lossFunction(classWeight).cuda()
                torch_dataset_train = Data.TensorDataset(torch.tensor(x_train),
                                                         torch.tensor(y_train).squeeze())
                data_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=False,
                                              drop_last=True,num_workers=1)

                test_dataset = Data.TensorDataset(torch.tensor(x_test),
                                                  torch.tensor(y_test).squeeze())
                test_dataLoader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                              drop_last=True, num_workers=1)

                # 可解释反馈训练
                model = trainModle_interpret(model, loss_func, optimizer, data_loader, EPOCH, index, test_dataLoader, train_stage=2)
            else:
                print('cant find gpu!!!')

            # 一折
            break
            # 分别读取模型
            ftcnn_state = {k[len("feature_process."):]: v for k, v in model.state_dict().items() if k.startswith("feature_process.")}
            classhead_state = {k[len("classifier."):]: v for k, v in model.state_dict().items() if k.startswith("classifier.")}
            ftcnn.load_state_dict(ftcnn_state)
            classhead.load_state_dict(classhead_state)

            # --- 准备数据 ---
            # 生成示例背景数据和待解释数据（实际项目中请替换为真实数据）
            # 背景数据：例如 100 个样本，每个样本形状 (1,3000)
            background_data = x_test[0:5]
            test_data = x_test[5:55]
            sample_num = 250

            background_data = ftcnn(x_test.cuda())
            test_data = ftcnn(x_test.cuda())

            # 转换成 torch tensor
            background_tensor = background_data
            test_tensor = test_data
            print("background and test: ", background_data.shape, test_data.shape)

            # --- 使用 SHAP 进行解释 ---
            # 输入模型1处理的结果，用模型2分类，获取每个特征贡献度
            explainer = shap.DeepExplainer(classhead.cpu(), background_tensor.cpu())
            # 计算测试数据的 SHAP 值，这里返回的 shap_values 是一个列表，
            # 列表中每个元素对应模型输出中一个类别的贡献，形状为 (10,1,3000)
            print(background_tensor.device, test_tensor.device)
            # test_tensor = test_tensor.clone().detach().requires_grad_(True)
            print("Test tensor requires_grad:", test_tensor.requires_grad)
            shap_values = explainer.shap_values(test_tensor.cpu())

            # 打印各类别 SHAP 值的形状信息
            print("SHAP values shapes:")
            for i, sv in enumerate(shap_values):
                # sv = sv.mean(axis=2)
                print(f"Class {i}: shape: {sv.shape}")

            # ---------------------------
            # 5. 聚合 SHAP 值并展示每个类别的解释结果
            # ---------------------------
            n_classes = 5  # 输出类别数
            n_bands = 5  # 频带数

            # 创建一行 5 个子图，每个子图对应一个类别
            fig, axes = plt.subplots(1, n_classes, figsize=(20, 5))
            if n_classes == 1:
                axes = [axes]

            # 针对每个类别计算各频带聚合的 SHAP 值
            for cls in range(n_classes):
                # 删除通道维度，转换为 (50, 5, 80)
                cls_shap = shap_values[cls]

                aggregated_shap = []  # 存放当前类别每个频带的平均绝对 SHAP 值

                for band in range(n_bands):
                    # 取出当前类别每个样本中频带 band 的 128 个 SHAP 值，形状 (50, 80)
                    band_shap = cls_shap[:, band, :]
                    # 对每个样本取 128 个数据的绝对值之和
                    band_importance = np.sum(np.abs(band_shap), axis=1)
                    # 对所有样本取平均
                    aggregated_shap.append(np.mean(band_importance))

                ax = axes[cls]
                ax.bar([f'{band_name[i]}' for i in range(n_bands)], aggregated_shap, color='steelblue')
                ax.set_xlabel('频带')
                ax.set_ylabel('平均绝对SHAP值')
                ax.set_title(f'类别 {cls} 的解释结果')

            plt.tight_layout()
            plt.show()

            del x_train
            del x_test
            del y_train
            del y_test
            del torch_dataset_train
            del data_loader
            del test_dataset
            del test_dataLoader
            del model
            gc.collect()

        index += 1
        print("current fold: ", index)