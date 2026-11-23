import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch

DATA_PATH = '/Users/ruyiyang/Documents/courseware/2026_Spring/mas60/mmai_project/hw2/affect_data/sarcasm.pkl'

traindata, validdata, test_robust = get_dataloader(DATA_PATH, robust_test=False, data_type='sarcasm')

# sarcasm modality dims: text=300, audio=81, video=371
encoders = [
    GRU(371, 512, dropout=True, has_padding=False, batch_first=True),
    GRU(81,  256, dropout=True, has_padding=False, batch_first=True),
    GRU(300, 600, dropout=True, has_padding=False, batch_first=True),
]
head = MLP(1368, 512, 1)

fusion = Concat()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoders = [e.to(device) for e in encoders]
head = head.to(device)
fusion = fusion.to(device)

train(
    encoders, fusion, head,
    traindata, validdata,
    total_epochs=50,
    task='classification',
    optimtype=torch.optim.AdamW,
    early_stop=True,
    is_packed=False,
    lr=1e-3,
    save='sarcasm_lf_best.pt',
    weight_decay=0.01,
    objective=torch.nn.CrossEntropyLoss(),
)

print("\nTesting:")
model = torch.load('sarcasm_lf_best.pt').to(device)
test(
    model=model,
    test_dataloaders_all=test_robust,
    dataset='sarcasm',
    is_packed=False,
    criterion=torch.nn.CrossEntropyLoss(),
    task='classification',
    no_robust=True,
)
