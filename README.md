# Deep_learning_0_2_1
In this project we are going to implement deep learning project step by step
Download dataset from ![link](https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar/download?datasetVersionNumber=2)

# 完成这些项目：
1. 图像分类
    - 简单多分类
    - 复杂多分类（数据不平衡，噪音）
2. 图像分割
3. 目标识别
4. 自然语言处理（文本分类）
    - 情绪分类
5. 时间序列预测
    - 预测类问题
6. 生成模型（GAN,AE,VAE）

6. 完成**1**的基础上再进行如下安排：
   - 分布式训练：学习如何在多个GPU或分布式环境下训练深度学习模型（训练技巧）
   - 模型优化和压缩：研究模型剪枝、量化、知识蒸馏等技术，以提高模型的推理速度和减少内存占用（训练技巧）
   - AutoML：学习自动化机器学习技术，使用工具如AutoKeras、Google AutoML等进行模型选择和超参数优化（训练技巧）
   - 云服务：熟悉AWS、Google Cloud、Azure等云平台的AI服务，学习如何在云上部署和管理模型（工程和部署）
   - 边缘计算：研究边缘设备上的深度学习应用，使用TensorFlow Lite、ONNX等框架在移动设备和物联网设备上部署模型（工程和部署）
   - 持续集成和持续部署（CI/CD）：学习如何在机器学习项目中实现CI/CD，使用工具如GitHub Actions、Jenkins等进行自动化部署和测试 （工程和部署）


# Git, DVC, 以及数据库介绍

## 数据库
我用的 [Google Drive](https://drive.google.com/drive/folders/1NE2MCMWE6OlvFni-B71KC4zwO8vEsr7d?usp=drive_link) 作为我们的远程数据库，用于储存所有的 `img` 文件、`model.pth` 和其他一些大型数据。储存在谷歌云中的三分类文件夹的 ID 地址是：`1TZ-RKDRaU4iwbZaDrIEOTe5TBvhnlCJH`。以后我们的数据都将存放在这个位置。

## DVC (Data Version Control) 和 Git
DVC 用于模型开发中的数据版本控制，它与 Git 一起使用，可以更好地管理和追踪数据的变化。以下是一些常用命令：

### DVC 命令
- `dvc add your_file`：添加文件到本地 DVC 仓库
- `dvc commit`：将当前更改记录到 DVC 缓存
- `dvc push`：将本地缓存的数据推送到远程存储
- `dvc pull`：从远程存储拉取数据到本地缓存
- `dvc checkout`：切换到指定的数据版本

### Git 命令
- `git add`：将文件添加到 Git 暂存区
- `git commit -m "xxxx"`：提交更改并添加注释
- `git push origin main` 或 `git push origin your_branch`：将更改推送到远程仓库
- `git status`：查看当前仓库状态
- `git log`：查看提交历史

### 示例 1
以下是一个使用 DVC 和 Git 进行数据版本控制的完整示例：

1. 初始化 DVC 项目：
    ```sh
    dvc init
    dvc remote add -d myremote 1TZ-RKDRaU4iwbZaDrIEOTe5TBvhnlCJH
   如果是第一次还需要将你的google cloud的帐号密码添加上
    dvc remote modify myremote gdrive_client_id <你的google cloud id>
    dvc remote modify myremote gdrive_client_secret <你的google cloud secret>
    ```

2. 添加数据文件到 DVC：
    ```sh
    dvc add data
    git add data.dvc .gitignore
    git commit -m "Add data version 1"
    ```

3. 推送到远程仓库：
    ```sh
    dvc commit
    dvc push
    git push origin main
    ```

4. 从远程拉取仓库：
    ```sh
    git clone git@github.com:Charly168/Deep_learning_0_2_1.git
   或者如果已经在仓库下就使用
    git pull (git pull origin main)
   你如果在develop分支的话，git pull 就会拉取develop分支，也就是自己的分支。你如果要拉取我的主分支就得git pull origin main
   
    
    ```

5. 推送到远程 Git 仓库：
    ```sh
    git push origin main
    ```

通过这些步骤，你可以确保数据文件和版本控制在 DVC 和 Git 中都保持同步，并能够追踪和管理数据的不同版本。

### 示例 2
以下是一个使用 DVC 和 Git 进行新分支的创建（一个你自己的分支，不能和主分支main混淆，不然如果发生一些误操作很麻烦）：
1. 新建分支并自动跳转到新分支develop：
    ```sh
    git checkout -b develop
    ```
2. 检查分支:
    ```sh
    git branch # 确保在develop分支
    ```
以后你就在deveop分支进行模型开发，在每次开发之前我建议你都使用 *git pull origin main* 将main 分支的最新内容都拉取下来。
然后使用*git merge main *
