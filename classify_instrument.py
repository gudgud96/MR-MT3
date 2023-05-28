import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

"""
Doesn't work. Instrument classification directly on encoder output is only 0.6 acc.
"""


class SimpleDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128)
        )
    def forward(self, x):
        return self.lin(x)


class InstEmbeddingDataset(Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode
        if mode == "train":
            self.keys = glob.glob("inst_test/train/*.npy")
            self.keys = list(set([k.split("/")[-1].split("_")[0] for k in self.keys]))
        elif mode == "validation":
            self.keys = glob.glob("inst_test/validation/*.npy")
            self.keys = list(set([k.split("/")[-1].split("_")[0] for k in self.keys]))
        elif mode == "test":
            self.keys = glob.glob("inst_test/test/*.npy")
            self.keys = list(set([k.split("/")[-1].split("_")[0] for k in self.keys]))
                
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = np.load(f"inst_test/{self.mode}/{key}_dec.npy")[1]
        label = np.load(f"inst_test/{self.mode}/{key}_ins.npy")
        if np.sum(label) == 0:
            return None, None
        return feature, label


def collate_fn(lst):
    x = [torch.from_numpy(k[0]) for k in lst if not k[0] is None]
    y = [torch.from_numpy(k[1]) for k in lst if not k[1] is None]
    return torch.stack(x), torch.stack(y)


def get_acc(label_out, label_gt):
    intersect = label_out.cpu().int() & label_gt.cpu().int()
    acc = torch.sum(intersect, dim=-1) / torch.sum(label_gt.cpu(), dim=-1)
    return acc
        

ds = InstEmbeddingDataset(mode="train")
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_ds = InstEmbeddingDataset(mode="validation")
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
criterion = nn.BCEWithLogitsLoss()
model = SimpleDense()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_val_loss = 10000
for ep in range(2000):
    avg_loss = 0
    avg_f1 = torch.tensor([])
    for item in dl:
        enc_out, label = item
        enc_out, label = enc_out.cuda().float(), label.cuda().float()
        optimizer.zero_grad()
        logits = model(enc_out)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        logits_sigmoid = torch.sigmoid(logits)
        logits_sigmoid[logits_sigmoid > 0.5] = 1
        logits_sigmoid[logits_sigmoid <= 0.5] = 0
        tmp = get_acc(logits_sigmoid, label)
        avg_f1 = torch.cat([
            avg_f1, tmp
        ])

        avg_loss += loss.item()
    
    avg_loss /= len(dl)
    avg_f1 = torch.mean(avg_f1).item()
   
    avg_val_loss = 0
    avg_val_f1 = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for item in val_dl:
            enc_out, label = item
            enc_out, label = enc_out.cuda().float(), label.cuda().float()
            logits = model(enc_out)
            loss = criterion(logits, label)
            avg_val_loss += loss.item()

            logits_sigmoid = torch.sigmoid(logits)
            logits_sigmoid[logits_sigmoid > 0.5] = 1
            logits_sigmoid[logits_sigmoid <= 0.5] = 0

            tmp = get_acc(logits_sigmoid, label)
            avg_val_f1 = torch.cat([
                avg_val_f1,
                tmp
            ])
    
    avg_val_loss /= len(val_dl)
    avg_val_f1 = torch.mean(avg_val_f1).item()
    model.train()
    
    print("epoch: {} loss: {:.4} acc: {:.4} val_loss: {:.4} val_acc: {:.4}".format(
        ep, avg_loss, avg_f1, avg_val_loss, avg_val_f1
    ))
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "inst.pt")
    


