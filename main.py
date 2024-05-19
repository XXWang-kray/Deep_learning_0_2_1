import torch.types
import os
from config import Config
from sklearn.model_selection import StratifiedShuffleSplit
from transform import Compose,ToTensor,RandomHorizontalFlip,RandomVerticalFlip,Resize,Normalize
from mydataset import Mydataset
from model import Alexnet

# TODO: 搭建模型(Alexnet,VGG,Resnet)
def createModel(device):

    model = Alexnet(ch_in=3,cls_num=3)
    model = model.to(device=device)

    return model


# TODO: 构建dataset（train+val+test）
def readDatalist(root,className):
    nameDic = {}
    images = []
    labels = []
    with open(className, 'r') as files:
        names = files.read().strip().split()
        for i, name in enumerate(names):
            nameDic[str(name)] = i

    for case in sorted(os.listdir(root)):
        imges = sorted(os.listdir(os.path.join(root, case)))
        for img in imges:
            images.append(os.path.join(os.path.join(root, case), img))
            labels.append(nameDic[case])

    return images, labels


# TODO: 数据预处理(数据增强，去噪等)

def main():
    arg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    imagesPath,labelsPath = readDatalist(arg.trainRoot,arg.className)
    testsImgPath,testsLabelPath = readDatalist(arg.testRoot,arg.className)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=arg.factor, random_state=42)
    train_indices, val_indices = next(sss.split(imagesPath, labelsPath))

    train_images = [imagesPath[i] for i in train_indices]
    train_labels = [labelsPath[i] for i in train_indices]

    val_images = [imagesPath[i] for i in val_indices]
    val_labels = [labelsPath[i] for i in val_indices]


    train_files = [{"image": image, "label": label} for image, label in zip(train_images, train_labels)]
    val_files = [{"image": image, "label": label} for image, label in zip(val_images, val_labels)]
    test_files = [{"image": image, "label": label} for image, label in zip(testsImgPath, testsLabelPath)]

    train_transform = Compose([
        Resize((128,128),interpolation='bilinear'),
        ToTensor(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        Normalize(),
    ])

    train_dataset = Mydataset(train_files,train_transform)
    val_dataset = Mydataset(val_files,train_transform)
    test_dataset = Mydataset(test_files,train_transform)

    model = createModel()



# TODO: 加载数据

# TODO： train,val,test

# TODO: 结果分析(acc,precision,recall)


if __name__ == "__main__":
    main()