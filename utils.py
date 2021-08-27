import os
import cv2
import numpy as np
import torch
import zipfile
import matplotlib.pyplot as plt
from termcolor import colored
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

#####################################################################################################

os.system("color")
def myprint(text, color) -> None:
    print(colored(text, color))


def breaker(num=50, char="*") -> None:
    myprint("\n" + num*char + "\n", "magenta")

#####################################################################################################

def unzip(path: str) -> None:
    with zipfile.ZipFile("./FMEyes.zip", 'r') as zip_ref:
        zip_ref.extractall(path)
    

def get_images(path: str, size: int) -> np.ndarray:
    images = np.zeros((len(os.listdir(path)), size, size, 3)).astype("uint8")
    i = 0
    for name in os.listdir(path):
        image = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
        images[i] = image
        i += 1
    return images


def save_graphs(L: list, A: list) -> None:
    TL, VL, TA, VA = [], [], [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graph")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Accuracy Graph")
    plt.savefig("./Graphs.jpg")
    plt.close("Plots")

#####################################################################################################

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEA_TRANSFORM = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
LOCAL_DATA_PATH = "./FMEyes/Train"                                                       
COLAB_DATA_PATH = "/content/FMEyes/Train"
LOCAL_TEST_DATA_PATH = ".FMEyes/Test" 
COLAB_TEST_DATA_PATH = "/content/FMEyes/Test"
CHECKPOINT_PATH = "./Checkpoints"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

#####################################################################################################

PRETRAINED_SIZE = 224