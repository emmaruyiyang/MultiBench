import torch
import sys
import os

sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from unimodals.common_models import Transformer, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat

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
    # Transformer(indim, outdim) — outputs last timestep hidden state
    encoders = [
        Transformer(371, 400).to(device),
        Transformer(81,  100).to(device),
        Transformer(300, 600).to(device),
    ]
    # 400 + 100 + 600 = 1100
    head = MLP(1100, 256, 2).to(device)  # 2-class: sarcastic / not sarcastic
    fusion = Concat().to(device)

    train(encoders, fusion, head, traindata, validdata, 50,
          task='classification',
          optimtype=torch.optim.AdamW,
          early_stop=True,
          is_packed=False,
          lr=1e-4,
          save='sarcasm_lf_transformer_best.pt',
          weight_decay=0.01,
          objective=torch.nn.CrossEntropyLoss())

    print("Testing:")
    model = torch.load('sarcasm_lf_transformer_best.pt', map_location=device)
    test(model=model, test_dataloaders_all=testdata, dataset='sarcasm',
         is_packed=False, criterion=torch.nn.CrossEntropyLoss(),
         task='classification', no_robust=True)
