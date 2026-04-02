import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image

# --- “识别图片大脑”蓝图 ---
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()

        # --- 卷积层 ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(p=0.25)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

        # --- 决策层 ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_fc = nn.Dropout(p=0.5)
        self.ceo_layer = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # --- 视觉流 ---

        # 卷积层 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        # 卷积层 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        # 卷积层 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # --- 决策流 ---
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop_fc(x)  # 决策前执行 Dropout
        x = self.ceo_layer(x)
        return x

# --- 数据处理器 ---
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_inference = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

CLASSES = ('飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')


# --- 定义“训练”的完整流程 (函数体不变，依赖主函数中的 fix) ---
def run_training(model, device, brain_file):
    # --- 训练参数定义 ---
    num_epochs = 1500  # 总轮数不变
    YOUR_TARGET_ACCURACY = 80.0

    # --- Checkpoint 文件名定义 ---
    # Checkpoint 文件用于临时保存，防止中断
    BRAIN_CHECKPOINT_FILE = brain_file.replace('.pth', '_checkpoint.pth')
    START_EPOCH = 0  # 默认从第0轮开始

    # --- 加载数据 ---
    print("--- 正在加载数据... ---")
    num_workers_to_use = 0 if device.type == 'cpu' else 4
    pin_memory_to_use = True if device.type == 'cuda' else False

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=num_workers_to_use,
                             pin_memory=pin_memory_to_use)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=num_workers_to_use,
                            pin_memory=pin_memory_to_use)
    print(f"--- “数据”加载完毕 (num_workers={num_workers_to_use}, pin_memory={pin_memory_to_use}) ---")

    # --- 准备工具，实例化 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 自动加载上次的训练成果或 Checkpoint ---
    load_file = None
    if os.path.exists(brain_file):
        load_file = brain_file
        print(f"--- 发现最终大脑文件 ({brain_file})，准备加载... ---")
    elif os.path.exists(BRAIN_CHECKPOINT_FILE):
        load_file = BRAIN_CHECKPOINT_FILE
        print(f"--- 发现 Checkpoint 文件 ({BRAIN_CHECKPOINT_FILE})，准备断点续训... ---")

    if load_file:
        checkpoint = torch.load(load_file, map_location=device)

        # 1. 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 2. 加载优化器状态 (保证学习率和动量是正确的)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 3. 加载调度器状态 (保证学习率衰减是正确的)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 4. 设置起始轮数 (从中断的地方继续)
        START_EPOCH = checkpoint['epoch']

        print(f"--- 专家大脑加载完毕！将从第 {START_EPOCH + 1} 轮继续训练... ---")
    else:
        print(f"--- 未发现“专家大脑”。将从第 1 轮开始“全新训练”... ---")

    # --- “大循环”开始 (从 START_EPOCH 开始) ---
    for epoch in range(START_EPOCH, num_epochs):
        current_epoch = epoch + 1  # 当前显示的轮数
        print(f"\n--- 第 {current_epoch} 遍“训练”开始 (共 {num_epochs} 遍) ---")

        model.train()
        running_loss = 0.0
        log_interval = 400

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # --- AI学习的“五句真言” ---
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # --- 训练记录 ---
            running_loss += loss.item()
            if i % log_interval == (log_interval - 1):
                print(f'[第 {current_epoch} 遍, 第 {i + 1:5d} 批] 平均错误率 (Loss): {running_loss / log_interval:.3f}')
                running_loss = 0.0
        print("--- “训练”结束 ---")

        # “调速器”介入
        scheduler.step()
        print(f"--- “调速器”已介入，下一轮“纠错力度” (lr) 调整为: {optimizer.param_groups[0]['lr']:.6f} ---")

        # 测试
        print("--- 开始测试“准确率”... ---")
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'--- 测试结束：本轮准确率: {accuracy:.2f} % ---')

        # --- Checkpointing: 打包所有状态 ---

        # 1. 检查点保存 (每 总数/10 轮保存一次临时记录)
        if (current_epoch) % 10 == 0:
            checkpoint = {
                'epoch': current_epoch,  # 保存当前轮数
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, BRAIN_CHECKPOINT_FILE)
            print(f"--- [CHECKPOINT] 第 {current_epoch} 轮临时记录已保存到 {BRAIN_CHECKPOINT_FILE} ---")

        # 判断 (Early Stopping)
        if accuracy >= YOUR_TARGET_ACCURACY:
            print(f"*** 恭喜！准确率 ({accuracy:.2f}%) 已达到您的预期 ({YOUR_TARGET_ACCURACY}%)！***")
            print("--- “提前停止” (Early Stopping) 训练！ ---")
            break
        else:
            print(f"--- 未达预期 (目标 {YOUR_TARGET_ACCURACY}%)，继续“训练”... ---")

    print('\n--- 最终训练完成！ ---')

    # 保存“专家大脑” (最终文件)
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(final_checkpoint, brain_file)
    print(f"--- “最终专家大脑”已保存到 {brain_file} ---")

    # 删除 Checkpoint 文件以保持工作区整洁
    if os.path.exists(BRAIN_CHECKPOINT_FILE):
        os.remove(BRAIN_CHECKPOINT_FILE)
        print("--- Checkpoint 文件已清理 ---")


# --- 定义“识别”的“完整流程” ---
def run_inference(model, device, brain_file):
    print("--- 正在启动“识别”模式 ---")

    # --- 加载“专家大脑” ---
    if not os.path.exists(brain_file):
        print(f"*** 错误：找不到“专家大脑”文件 ({brain_file})！ ***")
        print("--- 必须先运行“模式 1 (训练)”来“生成”一个“大脑”！ ---")
        return

    print(f"--- 正在加载“专家大脑” ({brain_file})... ---")
    # 1. 先加载完整的检查点字典
    checkpoint = torch.load(brain_file, map_location=device)

    # 2. 从字典中提取真正的模型状态字典 (model_state_dict)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("--- “专家大脑”加载完毕 ---")

    # --- 开启“推理模式” ---
    model.eval()

    # --- 接入需要识别的图片文件夹---
    PREDICT_DIR = './predict_images'
    if not os.path.exists(PREDICT_DIR):
        os.makedirs(PREDICT_DIR)
        print(f"--- (提示) 已自动创建文件夹: {PREDICT_DIR} ---")

    print(f"--- “待识别”文件夹: {PREDICT_DIR} ---")

    image_found = False

    # --- 遍历文件夹 ---
    for filename in os.listdir(PREDICT_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_found = True
            image_path = os.path.join(PREDICT_DIR, filename)

            try:
                # 加载并处理图片
                image_pil = Image.open(image_path).convert('RGB')

                # 处理图片 (强行缩放, 转Tensor, 归一化)
                image_tensor = transform_inference(image_pil)

                # 训练是按照批次的，所以需要将文件夹图片“升维”：(add a batch dimension) [3, 32, 32] -> [1, 3, 32, 32]
                image_batch = image_tensor.unsqueeze(0).to(device)

                # 执行推理
                with torch.no_grad():
                    # “大脑”猜答案 [1, 10]
                    outputs = model(image_batch)

                    # 翻译答案
                    _, predicted_index = torch.max(outputs.data, 1)
                    # 通过文档将数字转换成文字
                    predicted_class = CLASSES[predicted_index[0]]

                    print(f"  > 识别图片: '{filename}' ... 结果: 【{predicted_class}】")

            except Exception as e:
                print(f"  ! (错误) 处理图片 '{filename}' 失败: {e}")

    if not image_found:
        print(f"--- (提示) “{PREDICT_DIR}” 文件夹是“空”的。---")
        print(f"--- 请放入 .jpg/.png/.webp 图片，然后重新运行“模式 2”。 ---")

    print("--- 识别完成！ ---")


# --- “主菜单” (程序的“入口”) ---
if __name__ == "__main__":

    # 强制设置 'spawn' 启动方法，解决 L131 的稳定性问题
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

    # --- 准备“大脑”和“环境” ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化“空白”大脑
    model = DeeperCNN().to(device)
    # 大脑文件
    BRAIN_FILE = 'my_cifar10_brain_v4.pth'

    # --- 菜单 ---
    print("=" * 40)
    print("  欢迎使用 CIFAR-10 “二合一”AI工具")
    print("=" * 40)
    print(f"  “大脑”已部署到: {device}")
    print(f"  “大脑”文件: {BRAIN_FILE}")

    while True:
        print("\n请选择您的“工作模式”：")
        print("  (1) 训练模式 (继续学习/生成新大脑)")
        print("  (2) 识别模式 (扫描文件夹，获取识别结果)")
        print("  (3) 退出程序")

        mode = input("请输入 (1), (2) 或 (3): ")

        if mode == '1':
            # --- 训练模式 ---
            run_training(model, device, BRAIN_FILE)
            continue

        elif mode == '2':
            # --- 识别模式 ---
            run_inference(model, device, BRAIN_FILE)
            continue

        elif mode == '3':
            # --- 退出 ---
            print("程序退出。")
            break

        else:
            print("输入无效。请重新输入 (1), (2) 或 (3)。")