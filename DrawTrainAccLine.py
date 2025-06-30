import matplotlib.pyplot as plt

def read_floats_from_txt(file_path):
    """
    从文本文件中逐行读取浮点数，并返回一个列表
    每行应只包含一个可以转换为 float 的值
    """
    floats = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                value = float(line)
                floats.append(value)
            except ValueError:
                print(f"警告：无法将这一行转换为浮点数，已跳过 -> {line}")
    return floats

def plot_line(values):
    """
    根据一个数值列表绘制折线图
    """
    x = list(range(1, len(values) + 1))  # 横坐标从 1 开始
    y = values

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='-', color='tab:blue', label='Value')
    plt.title('Train val acc curve')
    plt.xlabel('epoch')
    plt.ylabel('val acc')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = './result/train_test_acc_baseline.txt'      # 将 data.txt 放在脚本同目录，或改为绝对路径
    values = read_floats_from_txt(file_path)
    if values:
        plot_line(values)
    else:
        print('未读取到文件内容')
