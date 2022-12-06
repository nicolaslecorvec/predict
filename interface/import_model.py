import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import stats
import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import time


model_name = 'tf_efficientnet_b2_ns'
device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()

def import_model():
    model = torch.jit.load('/content/drive/MyDrive/Colab Notebooks/model_scripted.pt')
    model.eval()
    return model
