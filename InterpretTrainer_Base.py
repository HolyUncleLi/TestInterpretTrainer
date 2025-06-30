from sklearn.preprocessing import StandardScaler
import pickle
from processData import *
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.nn.functional as F
# from TCNSleepNet_2 import *
import warnings
from scipy.interpolate import interp1d
import logging


Batch_Size = 64
seq_len = 1

warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()   # 判断GPU是否存在可用
keys = ["Fpz-Cz", "Pz-Oz", "label"]
result_path = "./result/"


# Normalizing and return t he normalized data and standardscaler
def standardScalerData(standardscaler, x_data):
    standardscaler.fit(x_data)
    x_standard = standardscaler.transform(x_data)
    return torch.from_numpy(x_standard), standardscaler


features = []
def hook_fn(module, input, output):
    features.append(output)


def grad_cam(model, x):
    gradients = []

    def save_gradient(grad):
        gradients.append(grad)

    x = x.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    # x.requires_grad = True
    output = model(x.view(Batch_Size, 1, 3000))

    # 反向传播
    model.zero_grad()
    target = output[0, torch.argmax(output[0])]
    target.backward()
    # print("predict label: ", target)

    # 获取梯度和特征图
    gradients = model.embed.weight.grad.data.cpu().numpy()[0]
    # print("grad shape: ", gradients.shape)
    # print("features shape: ", len(features), features[-1].shape)
    features_mean = features[-1].mean(dim=1).detach().cpu().numpy()
    # print("features mean shape: ", len(features), features_mean.shape)

    # weights = np.mean(gradients, axis=1)
    weights = gradients[np.newaxis,:]
    # print("weights shape: ",weights.shape)
    # cam = np.sum(weights[:, np.newaxis] * x.detach().numpy()[0, 0], axis=0)
    '''
    cam = np.zeros([80])
    for i in range(80):
        print(weights[i], features_mean[:, i].shape)
        cam += weights[i] * features_mean[:, i]
    '''
    cam = weights * features_mean
    # print(weights[0].shape, ( weights[0] * features[-1][0,0].detach().numpy()).shape, features[-1][0,0].detach().numpy().shape)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # 假设 cam 的形状为 (64, 80)
    original_array = cam

    # 这里针对第二个维度，原始索引长度为 80
    original_indices = np.linspace(0, 1, original_array.shape[1])
    target_indices = np.linspace(0, 1, 3000)

    # print("Original indices shape:", original_indices.shape, "Original array shape:", original_array.shape)

    # 沿着 axis=1 进行插值，每一行的数据将由 80 个点扩展为 3000 个点
    interp_function = interp1d(original_indices, original_array, kind='linear', axis=1)
    cam = interp_function(target_indices)

    # print("Final cam shape:", cam.shape)  # 预期输出形状为 (64, 3000)
    return cam


# -------------------------------
# 定义包含解释正则项的自定义损失函数
# -------------------------------
def custom_loss(model, x, target, A, lambda_explanation=1000.0):
    """
    参数：
      model             : 神经网络模型
      x                 : 输入张量，形状 (batch_size, input_dim)，需设置 requires_grad=True
      target            : 真实标签，形状 (batch_size,)
      A                 : 注释掩码，形状 (batch_size, input_dim)，二值张量，1 表示该特征“不应该”对预测敏感
      lambda_explanation: 正则项权重，用于平衡交叉熵损失与解释损失

    返回：
      loss_total        : 总损失 = cross-entropy loss + explanation regularization loss
      loss_ce           : 分类交叉熵损失
      loss_explanation  : 解释正则（梯度惩罚）项损失
    """
    # 前向传递：得到预测的 logits
    logits = model(x.view(Batch_Size, seq_len, 3000))  # (batch_size, num_classes)

    # 计算交叉熵损失（注意：PyTorch 的 cross_entropy 中内置了 softmax）
    loss_ce = F.cross_entropy(logits, target)

    # 计算 log-softmax 得到对数概率
    log_probs = F.log_softmax(logits, dim=1)  # (batch_size, num_classes)

    # 为了获得稳健性，论文中建议将各类别的 log 概率求和后再对输入求梯度
    # 得到形状 (batch_size,) 的每个样本的标量结果
    sum_log_probs = log_probs.sum(dim=1)

    # 计算 sum_log_probs 对输入 x 的梯度，设置 create_graph=True 以便对正则项二阶求导
    gradients = torch.autograd.grad(
        outputs=sum_log_probs,
        inputs=x,
        grad_outputs=torch.ones_like(sum_log_probs),
        create_graph=True,
        retain_graph=True
    )[0]  # 形状为 (batch_size, input_dim)

    # 解释正则项：在 A 指定为 1 的特征上惩罚梯度的大值
    # 这里对梯度取平方后乘以 A，然后对 batch 内所有元素求和
    # print(lambda_explanation, A.shape, gradients.shape, gradients.pow(2).shape, x.size(0))
    loss_explanation = lambda_explanation * (A * gradients.pow(2)).sum() / x.size(0)

    # 总损失为交叉熵损失和解释正则项之和
    loss_total = loss_ce + loss_explanation
    # loss_total = loss_ce

    return loss_total, loss_ce, loss_explanation


class lossFunction_interpret(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super(lossFunction_interpret, self).__init__()
        self.reduction = reduction
        self.w = weight

    def forward(self, model, x, target, A, lambda_explanation=1000.0):
        """
        参数：
          model             : 神经网络模型
          x                 : 输入张量，形状 (batch_size, input_dim)，需设置 requires_grad=True
          target            : 真实标签，形状 (batch_size,)
          A                 : 注释掩码，形状 (batch_size, input_dim)，二值张量，1 表示该特征“不应该”对预测敏感
          lambda_explanation: 正则项权重，用于平衡交叉熵损失与解释损失

        返回：
          loss_total        : 总损失 = cross-entropy loss + explanation regularization loss
          loss_ce           : 分类交叉熵损失
          loss_explanation  : 解释正则（梯度惩罚）项损失
        """
        # 前向传递：得到预测的 logits
        logits = model(x)  # (batch_size, num_classes)

        # 计算交叉熵损失（注意：PyTorch 的 cross_entropy 中内置了 softmax）
        loss_ce = F.cross_entropy(logits, target)

        # 计算 log-softmax 得到对数概率
        log_probs = F.log_softmax(logits, dim=1)  # (batch_size, num_classes)

        # 为了获得稳健性，论文中建议将各类别的 log 概率求和后再对输入求梯度
        # 得到形状 (batch_size,) 的每个样本的标量结果
        sum_log_probs = log_probs.sum(dim=1)

        # 计算 sum_log_probs 对输入 x 的梯度，设置 create_graph=True 以便对正则项二阶求导
        gradients = torch.autograd.grad(
            outputs=sum_log_probs,
            inputs=x,
            grad_outputs=torch.ones_like(sum_log_probs),
            create_graph=True,
            retain_graph=True
        )[0]  # 形状为 (batch_size, input_dim)

        # 解释正则项：在 A 指定为 1 的特征上惩罚梯度的大值
        # 这里对梯度取平方后乘以 A，然后对 batch 内所有元素求和
        A = torch.tensor(grad_cam(model, x)).unsqueeze(dim=1).cuda()
        # print(lambda_explanation, A.shape, gradients.pow(2).shape, x.size(0))
        loss_explanation = lambda_explanation * (A * gradients.pow(2)).sum() / x.size(0)

        # 总损失为交叉熵损失和解释正则项之和
        loss_total = loss_ce + loss_explanation

        return loss_total


#  带权损失函数
class lossFunction(torch.nn.Module):
    def __init__(self, weight, reduction='mean'):
        super(lossFunction, self).__init__()
        self.reduction = reduction
        self.w = weight

    def forward(self, logits, target):

        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]
        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)  # [NHW, 1]
        loss = -1 * logits

        for i in range(logits.shape[0]):
            loss[i] *= self.w[target[i]]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def trainModle_interpret(model, lossname, optimizername, data_loader, EPOCH, index, test_data, train_stage=1):

    model = model.cuda()
    best_model = model
    total_loss = 0
    best_epoch = 0
    maxAcc = 0

    hook = model.embed.register_forward_hook(hook_fn)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # 打终端
            logging.FileHandler("train.log"),  # 也写 train.log
        ]
    )

    for epoch in range(EPOCH):
        model.train()
        for step, (train_x, train_y) in enumerate(data_loader):
            train_x = train_x.cuda()
            train_x.requires_grad_(True)  # ← 关键：确保它可求导
            train_y = torch.squeeze(train_y.view(-1,)).type(torch.LongTensor).cuda()
            A = torch.randint(1, 2, (Batch_Size, 3000)).float().cuda()

            # loss, loss_ce, loss_explanation = custom_loss(model, train_x, train_y, A, lambda_explanation=1.0)
            logits = model(train_x.view(Batch_Size, seq_len, 3000))  # (batch_size, num_classes)
            loss = lossname(logits, train_y)

            total_loss += loss

            optimizername.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizername.step()

        # -------------------------------
        # 每 5 轮验证模型效果
        # -------------------------------
        if epoch % 5 == 0:
            model.eval()
            pred_y_list = []
            label_y_list = []
            for step, (test_x, test_y) in enumerate(test_data):
                test_x = test_x.cuda()
                test_y = torch.squeeze(test_y.view(-1,)).type(torch.LongTensor).cuda()

                model = model.cuda()
                output = model(test_x.view(Batch_Size, seq_len, 3000))

                test_pred_y = torch.max(output, 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y.cpu())

            report_acc = classification_report(pred_y_list, label_y_list, digits=6, output_dict=True)['accuracy']
            if report_acc > maxAcc:
                maxAcc = report_acc
                best_model = model
                best_epoch = epoch
                torch.cuda.empty_cache()
                torch.save(best_model.state_dict(), './result/models/%d.pth' % (index + 1))
                print('saved model')

            # print("index: ", index, " | epoch: ", epoch, " | val acc: ", report_acc, " | avg loss: ", )
            logging.info(
                f"idx {index:<2d} | "
                f"epoch {epoch:<3d} | "
                f"val_acc {report_acc:.4f} | "
                f"avg_loss {(total_loss/5).item():.4f}"
            )
            with open("./result/models/train_acc_curve.txt", "a", encoding="utf-8") as file:
                file.write(str(report_acc)+'\n')
            total_loss = 0

    model.eval()
    pred_y_list = []
    label_y_list = []
    right_pre_class = [0, 0, 0, 0, 0]
    for step, (test_x, test_y) in enumerate(test_data):
        # test_x, test_y = noshuffleData(test_x, test_y)
        # test_x, standardscaler = standardScalerData(standardscaler, test_x)

        model = model.cuda()
        # test_x = torch.unsqueeze(test_x, 1).type(torch.FloatTensor).cuda()
        test_x = test_x.cuda()
        test_y = torch.squeeze(test_y.view(-1,)).type(torch.LongTensor).cuda()

        # output = modelname(test_x.view(Batch_Size, 1, 3000).transpose(1,2), train_stage)
        output = model(test_x.view(Batch_Size, seq_len, 3000))

        test_pred_y = torch.max(output, 1)[1].cpu()
        pred_y_list.extend(test_pred_y)
        label_y_list.extend(test_y.cpu())
    report_acc = classification_report(pred_y_list, label_y_list, digits=6, output_dict=True)['accuracy']
    # print("val acc: ", report_acc," | loss: ", total_loss)
    if (report_acc > maxAcc):
        maxAcc = report_acc
        best_model = model
        best_epoch = EPOCH
    print("max acc : ", maxAcc, "best epoch : ", best_epoch)

    for i in range(len(pred_y_list)):
        if pred_y_list[i] == label_y_list[i]:
            right_pre_class[pred_y_list[i]] += 1


    total_num = checkDataset(label_y_list)
    for i in range(5):
        right_pre_class[i] = right_pre_class[i] / total_num[i]
    print("right pre-class:", right_pre_class)

    # Saving the datas
    if train_stage == 2:
        # save net
        torch.cuda.empty_cache()
        torch.save(best_model.state_dict(), './result/models/%d.pth' % (index + 1))

        f = open("./result/k.txt", "a")
        f.write(str(maxAcc) + '\n')
        f.close()

        f = open("./result/wake.txt", "a")
        f.write(str(right_pre_class[0]) + '\n')
        f.close()

        f = open("./result/n1.txt", "a")
        f.write(str(right_pre_class[1]) + '\n')
        f.close()

        f = open("./result/n2.txt", "a")
        f.write(str(right_pre_class[2]) + '\n')
        f.close()

        f = open("./result/n3.txt", "a")
        f.write(str(right_pre_class[3]) + '\n')
        f.close()

        f = open("./result/rem.txt", "a")
        f.write(str(right_pre_class[4]) + '\n')
        f.close()

    return best_model
