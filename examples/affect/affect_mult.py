import torch
import sys
import os

sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from fusions.mult import MULTModel
from unimodals.common_models import Identity
from datasets.affect.get_data import get_dataloader

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

    class HParams():
        num_heads = 4
        layers = 3
        attn_dropout = 0.1
        attn_dropout_modalities = [0, 0, 0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 2   # 2-class: sarcastic / not sarcastic
        all_steps = False

    # MulT handles encoding internally via Conv1d + cross-attention
    # encoders are Identity; n_features = raw modality dims [video, audio, text]
    encoders = [Identity().to(device), Identity().to(device), Identity().to(device)]
    fusion = MULTModel(3, [371, 81, 300], hyp_params=HParams).to(device)
    head = Identity().to(device)  # MulT has its own output layer

    train(encoders, fusion, head, traindata, validdata, 50,
          task='classification',
          optimtype=torch.optim.AdamW,
          early_stop=True,
          is_packed=False,
          lr=1e-3,
          clip_val=1.0,
          save='sarcasm_mult_best.pt',
          weight_decay=0.01,
          objective=torch.nn.CrossEntropyLoss())

    print("Testing:")
    model = torch.load('sarcasm_mult_best.pt', map_location=device)
    test(model=model, test_dataloaders_all=testdata, dataset='sarcasm',
         is_packed=False, criterion=torch.nn.CrossEntropyLoss(),
         task='classification', no_robust=True)
