import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score
import os

"""
instrument classification - 0.6 F1
pitch classification - 0.4 F1
"""

class SimpleDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(512, 16)

    def forward(self, x):
        return self.lin(x)
    
# validation set
val_keys = sorted(glob.glob("slakh_mt3_hidden_state_eval_validation/*/"))
test_keys = sorted(glob.glob("slakh_mt3_hidden_state_eval_test/*/"))

print("val keys:", len(val_keys), "test keys:", len(test_keys))

val_enc_hidden = []
val_instruments = []
val_pitches = []
for key in tqdm(val_keys):
    enc_hidden = np.load(os.path.join(key, "enc_hidden.npy"))
    instruments = np.load(os.path.join(key, "instruments.npy"))

    assert np.argwhere(instruments == 1).shape[0] > 0
    pitches = np.load(os.path.join(key, "pitches.npy"))

    val_enc_hidden.append(enc_hidden)
    val_instruments.append(instruments)
    val_pitches.append(pitches)

val_enc_hidden = np.concatenate(val_enc_hidden, axis=0)
val_instruments = np.concatenate(val_instruments, axis=0)
val_pitches = np.concatenate(val_pitches, axis=0)

np.save("val_enc_hidden.npy", val_enc_hidden)
np.save("val_instruments.npy", val_instruments)
np.save("val_pitches.npy", val_pitches)

print("val enc hidden:", val_enc_hidden.shape)
print("val instruments:", val_instruments.shape)
print("val pitches:", val_pitches.shape)

# test set
test_enc_hidden = []
test_instruments = []
test_pitches = []

for key in tqdm(test_keys):
    enc_hidden = np.load(os.path.join(key, "enc_hidden.npy"))
    instruments = np.load(os.path.join(key, "instruments.npy"))
    pitches = np.load(os.path.join(key, "pitches.npy"))

    test_enc_hidden.append(enc_hidden)
    test_instruments.append(instruments)
    test_pitches.append(pitches)

test_enc_hidden = np.concatenate(test_enc_hidden, axis=0)
test_instruments = np.concatenate(test_instruments, axis=0)
test_pitches = np.concatenate(test_pitches, axis=0)

np.save("test_enc_hidden.npy", test_enc_hidden)
np.save("test_instruments.npy", test_instruments)
np.save("test_pitches.npy", test_pitches)

print("test enc hidden:", test_enc_hidden.shape)
print("test instruments:", test_instruments.shape)
print("test pitches:", test_pitches.shape)


class InstEmbeddingDataset(Dataset):
    def __init__(self, type="instrument", mode="validation"):
        super().__init__()
        self.type = type
        if mode == "validation":
            self.enc_hidden = np.load("val_enc_hidden.npy")
            self.instruments = np.load("val_instruments.npy")
            self.pitches = np.load("val_pitches.npy")
        elif mode == "test":
            self.enc_hidden = np.load("test_enc_hidden.npy")
            self.instruments = np.load("test_instruments.npy")
            self.pitches = np.load("test_pitches.npy")
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.enc_hidden)

    def __getitem__(self, idx):
        if self.type == "enc":
            return self.enc_hidden[idx]
        elif self.type == "instrument":
            return self.enc_hidden[idx], self.instruments[idx]
        elif self.type == "pitch":
            return self.enc_hidden[idx], self.pitches[idx]
        else:
            raise NotImplementedError


def collate_fn(lst):
    x = [torch.from_numpy(k[0]) for k in lst]
    y = [torch.from_numpy(k[1]) for k in lst]
    return torch.stack(x), torch.stack(y)


def get_acc(label_out, label_gt):
    f1 = precision_score(
        label_gt.cpu().int().detach().numpy(), 
        label_out.cpu().int().detach().numpy(), 
        average='macro',
        zero_division=0
    )
    return f1
    
    # intersect = label_out.cpu().int() & label_gt.cpu().int()
    # acc = torch.sum(intersect, dim=-1) / torch.sum(label_gt.cpu(), dim=-1)
    # return torch.mean(acc)
        

type = "instrument"
ds = InstEmbeddingDataset(mode="validation", type=type)
dl = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_ds = InstEmbeddingDataset(mode="test", type=type)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
criterion = nn.BCEWithLogitsLoss()
model = SimpleDense()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_val_loss = 10000
for ep in range(2000):
    avg_loss = 0
    avg_f1 = 0
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
        f1 = get_acc(logits_sigmoid, label)
        avg_f1 += f1

        avg_loss += loss.item()
    
    avg_loss /= len(dl)
    avg_f1 /= len(dl)
   
    avg_val_loss = 0
    avg_val_f1 = 0
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