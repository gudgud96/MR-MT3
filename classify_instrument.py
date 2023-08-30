import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import pickle

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
    def __init__(self, mode="train", type="enc", head=8):
        super().__init__()
        self.mode = mode
        if mode == "train":
            self.keys = sorted(glob.glob("embeddings/train/*/mix.pickle"))
        elif mode == "validation":
            self.keys = sorted(glob.glob("embeddings/validation/*/mix.pickle"))
        elif mode == "test":
            self.keys = sorted(glob.glob("embeddings/test/*/mix.pickle"))
        
        self.type = type
        self.head = head
                
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with open(key, "rb") as f:
            dic = pickle.load(f)
        
        embedding_lst = []
        inst_labels = []
        for idx in range(len(dic)):
            embedding = dic[idx][self.type][self.head]
            embedding = np.mean(embedding, axis=0)
            embedding_lst.append(embedding)
            
            inst_label = np.zeros(128)
            is_drum = False
            for note in dic[idx]["event"].notes:
                if note.is_drum:
                    is_drum = True
                else:
                    inst_label[note.program] = 1

            inst_labels.append(inst_label)
        
        return np.stack(embedding_lst), np.stack(inst_labels)


def collate_fn(lst):
    x = [torch.from_numpy(k[0]) for k in lst]
    y = [torch.from_numpy(k[1]) for k in lst]
    return torch.cat(x), torch.cat(y)


def get_acc(label_out, label_gt):
    f1 = f1_score(label_gt.cpu().int().detach().numpy(), label_out.cpu().int().detach().numpy(), average='macro', zero_division=1)
    # print(f1)
    # intersect = label_out.cpu().int() & label_gt.cpu().int()
    # acc = torch.sum(intersect, dim=-1) / torch.sum(label_gt.cpu(), dim=-1)
    return f1
        

ds = InstEmbeddingDataset(mode="train", type="dec", head=0)
dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=12)
val_ds = InstEmbeddingDataset(mode="validation", type="dec", head=0)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
criterion = nn.BCEWithLogitsLoss()
model = SimpleDense()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_val_loss = 10000
for ep in range(2000):
    avg_loss = 0
    avg_f1 = 0
    for item in tqdm(dl):
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
        f1 = get_acc(logits_sigmoid, label)
        avg_f1 += f1

        avg_loss += loss.item()
    
    avg_loss /= len(dl)
    avg_f1 /= len(dl)
   
    avg_val_loss = 0
    avg_val_f1 = 0
    model.eval()
    with torch.no_grad():
        for item in tqdm(val_dl):
            enc_out, label = item
            enc_out, label = enc_out.cuda().float(), label.cuda().float()
            logits = model(enc_out)
            loss = criterion(logits, label)
            avg_val_loss += loss.item()

            logits_sigmoid = torch.sigmoid(logits)
            logits_sigmoid[logits_sigmoid > 0.5] = 1
            logits_sigmoid[logits_sigmoid <= 0.5] = 0

            tmp = get_acc(logits_sigmoid, label)
            avg_val_f1 += f1
    
    avg_val_loss /= len(val_dl)
    avg_val_f1 /= len(val_dl)
    model.train()
    
    print("epoch: {} loss: {:.4} f1: {:.4} val_loss: {:.4} val_f1: {:.4}".format(
        ep, avg_loss, avg_f1, avg_val_loss, avg_val_f1
    ))
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "inst.pt")
    


