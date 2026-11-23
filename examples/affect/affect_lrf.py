import torch
import sys
import os

sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import LowRankTensorFusion
from torch import nn

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    traindata, validdata, testdata = get_dataloader(
        'examples/affect/pack/MUStARD/sarcasm.pkl',
        robust_test=False,
        data_type='sarcasm',
        max_pad=True,
        max_seq_len=50,
        task='classification',
        batch_size=32,
    )

    # sarcasm: video=371, audio=81, text=300
    # GRU(last_only=True) → (batch, hiddim), then Linear projects to smaller dim
    encoders = [
        nn.Sequential(GRU(371, 512, dropout=True, has_padding=False, last_only=True, batch_first=True), nn.Linear(512, 32)).to(device),
        nn.Sequential(GRU(81,  256, dropout=True, has_padding=False, last_only=True, batch_first=True), nn.Linear(256, 32)).to(device),
        nn.Sequential(GRU(300, 600, dropout=True, has_padding=False, last_only=True, batch_first=True), nn.Linear(600, 128)).to(device),
    ]

    # input_dims must match encoder output dims [32, 32, 128]
    fusion = LowRankTensorFusion([32, 32, 128], 128, 32).to(device)
    head = MLP(128, 512, 2).to(device)  # 2-class: sarcastic / not sarcastic

    train(encoders, fusion, head, traindata, validdata, 50,
          task='classification',
          optimtype=torch.optim.AdamW,
          early_stop=True,
          is_packed=False,
          lr=1e-3,
          save='sarcasm_lrf_best.pt',
          weight_decay=0.01,
          objective=torch.nn.CrossEntropyLoss())

    print("Testing:")
    model = torch.load('sarcasm_lrf_best.pt', map_location=device)
    test(model=model, test_dataloaders_all=testdata, dataset='sarcasm',
         is_packed=False, criterion=torch.nn.CrossEntropyLoss(),
         task='classification', no_robust=True)
