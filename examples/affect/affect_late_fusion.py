import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch

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

    encoders = [
        GRU(371, 512, dropout=True, has_padding=False, last_only=True, batch_first=True).to(device),
        GRU(81,  256, dropout=True, has_padding=False, last_only=True, batch_first=True).to(device),
        GRU(300, 600, dropout=True, has_padding=False, last_only=True, batch_first=True).to(device),
    ]
    # 512 + 256 + 600 = 1368
    head = MLP(1368, 512, 2).to(device)
    fusion = Concat().to(device)

    train(encoders, fusion, head, traindata, validdata, 50,
          task='classification',
          optimtype=torch.optim.AdamW,
          early_stop=True,
          is_packed=False,
          lr=1e-3,
          save='sarcasm_lf_best.pt',
          weight_decay=0.01,
          objective=torch.nn.CrossEntropyLoss())

    print("Testing:")
    model = torch.load('sarcasm_lf_best.pt', map_location=device)
    test(model=model, test_dataloaders_all=testdata, dataset='sarcasm',
         is_packed=False, criterion=torch.nn.CrossEntropyLoss(),
         task='classification', no_robust=True)
