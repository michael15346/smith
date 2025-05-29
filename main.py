import sys
import os
import math
from pathlib import Path
from utils import lit2i
from const import PCAP_FILE_HEADER_LEN, PCAP_PACKET_HEADER_LEN
from encoder import Decoder, PCAP, IPv4, IPv6, TCP, UDP
import numpy as np
from torch.nn import Sequential, Linear, LSTM, ReLU, Conv2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, Sigmoid
from torch.optim import SGD
from torch import Tensor, squeeze, unsqueeze, nn, tensordot
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torch
import torch.nn.functional as F
from scipy.signal.windows import blackmanharris
import librosa
import soundfile as sf

BATCH_SIZE = 2048


def create_mel_filterbank(sr, n_fft, n_mels):
    """Returns a torch.Tensor of shape [n_mels, n_fft//2 + 1]"""
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return torch.tensor(mel_fb, dtype=torch.float32)


def collate_as_list(batch):
    tensors, tags = zip(*batch)  # tuples of tensors
    return list(tensors), list(tags)


class SlidingWindowMelFFT(nn.Module):
    def __init__(self, window_size, hop_size, sampling_rate=44100, n_mels=8, device='cuda'):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.device = device

        win = blackmanharris(window_size)#get_window(window_type, window_size, fftbins=True)
        self.register_buffer('window', torch.tensor(win, dtype=torch.float32).view(1, -1))  # [1, W]

        mel_fb = create_mel_filterbank(sr=sampling_rate, n_fft=window_size, n_mels=n_mels)
        self.register_buffer('mel_fb', mel_fb.to(device))  # [n_mels, F]

    def forward(self, signal_list):
        specs = []
        lengths = []

        for signal in signal_list:
            x = signal.to(self.device)
            if x.shape[0] < self.window_size:
                x = F.pad(x, (0, self.window_size - x.shape[0]))

            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            frames = x.unfold(-1, self.window_size, self.hop_size).squeeze(0).squeeze(0)  # [T, W]
            windowed = frames * self.window  # [T, W]
            fft_vals = torch.fft.rfft(windowed, dim=-1)  # [T, F]
            power = torch.abs(fft_vals) ** 2  # [T, F]
            mel_spec = power @ self.mel_fb.T  # [T, n_mels]
            mel_spec = mel_spec.T  # [n_mels, T]

            specs.append(mel_spec)
            lengths.append(mel_spec.shape[1])

        T_max = max(lengths)
        padded = [F.pad(s, (0, T_max - s.shape[1])) for s in specs]
        batch = torch.stack(padded, dim=0).unsqueeze(1)  # [B, 1, n_mels, T_max]
        return batch


def pad_to(x: Tensor, T_target: int):
    diff = T_target - x.data.shape[-1]
    if diff == 0:
        return x
    data = F.pad(x.data, (0, diff))
    return data


class UberModel(nn.Module):

    def __init__(self): 
        super().__init__()
        self.nb = SlidingWindowMelFFT(window_size=128, hop_size=32, sampling_rate=44100, device='cpu')
        self.wb = SlidingWindowMelFFT(window_size=1024, hop_size=256, sampling_rate=44100, device='cpu')

        self.c1 = Conv2d(1, 1, (3, 3))
        self.c2 = Conv2d(1, 1, (3, 3))
        self.cnn = Sequential(
                Conv2d(2, 4, (3, 3)),
                ReLU()
                )
        self.mp = AdaptiveMaxPool2d((1,1))
        self.ap = AdaptiveAvgPool2d((1,1))
        self.mlp = Sequential(
                Linear(4, 2),
                Linear(2, 4)
                )
        self.layerback = Linear(4, 1)
        self.cbam = Conv2d(2, 1, (7, 7), padding='same')
        self.sgm = Sigmoid()
        self.relu = ReLU()
        self.lstm = LSTM(4, 4, bidirectional=True)
        self.fc2 = Linear(8, 7)

    def forward(self, x):
        nb = self.nb(x)
        wb = self.wb(x)

        nb_conv = self.relu(self.c1(nb))
        wb_conv = self.relu(self.c2(wb))
        T_max = max(nb_conv.data.shape[-1], wb_conv.data.shape[-1])
        
        nb_conv = pad_to(nb_conv, T_max)
        wb_conv = pad_to(wb_conv, T_max)

        stacked = torch.cat([nb_conv.data, wb_conv.data], dim=1)
        s2 = self.cnn(stacked)
        s2_mp = self.mp(s2).squeeze((-1, -2))
        s2_ap = self.ap(s2).squeeze((-1, -2))
        v1 = self.mlp(s2_mp)
        v2 = self.mlp(s2_ap)
        mc = self.sgm(v1 + v2).unsqueeze(-1).unsqueeze(-1)
        fmc = s2 * mc
        print(fmc.shape)
        fmc_mp, _ = fmc.max(dim=1)
        fmc_mp = fmc_mp.unsqueeze(1)
        print(fmc_mp.shape)
        fmc_ap = fmc.mean(dim=1).unsqueeze(1)
        print(fmc_ap.shape)
        fmc_comb = torch.cat([fmc_mp, fmc_ap], dim=1)
        fmc_conv = self.cbam(fmc_comb)
        fmc_conv = self.sgm(fmc_conv)
        ff = fmc * fmc_conv
        mel_shape = ff.shape[2]
        samples = ff.shape[3]
        print(ff.shape)
        ff = ff.transpose(1,3)
        print(ff.shape)
        ff = self.layerback(ff)
        print(ff.shape)
        ff = ff.transpose(1, 3)
        print(ff.shape)
        ff = ff.squeeze(1)
        print(ff.shape)
        ff = ff.transpose(1,2)
        print(ff.shape)
        al, _ = self.lstm(ff)
        c = self.fc2(al[:, -1, :])
        


        return c



def eval_model(dataset, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (data, tag) in enumerate(dataset):
            inputs, targets = data, tag
            targets = torch.stack(targets)
            preds = model(inputs)
            #print(preds.shape)
            #print(targets.shape)
            # print(preds)
            # input()
            loss = nn.BCEWithLogitsLoss()(preds, targets)
            test_loss += loss.item()
    return test_loss


def preprocess(file_path):
    audio, sr = sf.read(file_path)
    return torch.tensor(audio, dtype=torch.float32)


class MultiHotFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []  # (file_path, folder_path)
        self.folder_to_tags = {}  # folder_path -> list of tag strings
        all_tags = set()

        # First pass: collect all tags and files
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            tag_file = os.path.join(folder_path, 'tag.txt')
            with open(tag_file, 'r') as f:
                tags = [line.strip() for line in f]
            self.folder_to_tags[folder_path] = tags
            all_tags.update(tags)

            for file_name in os.listdir(folder_path):
                if file_name == 'tag.txt':
                    continue
                file_path = os.path.join(folder_path, file_name)
                self.samples.append((file_path, folder_path))

        # Build tag-to-index mapping
        self.tag_to_index = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        self.num_tags = len(self.tag_to_index)

        # Precompute multi-hot tag tensors per folder
        self.folder_multi_hot = {
            folder: self._encode_tags(tags)
            for folder, tags in self.folder_to_tags.items()
        }

    def _encode_tags(self, tags):
        multi_hot = torch.zeros(self.num_tags, dtype=torch.float)
        for tag in tags:
            index = self.tag_to_index[tag]
            multi_hot[index] = 1.0
        return multi_hot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, folder_path = self.samples[idx]
        tensor = preprocess(file_path)
        tags_tensor = self.folder_multi_hot[folder_path]
        return tensor, tags_tensor

# Example usage
def main(argv):

    path = Path('../data')

    dataset = MultiHotFolderDataset("../data")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    datasets = random_split(dataset, (0.8, 0.1, 0.1))
    d_train = datasets[0]
    d_test = datasets[1]
    d_val = datasets[2]
    dld = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_as_list)
    dld_test = DataLoader(d_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_as_list)
    dld_val = DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_as_list)


    model = UberModel()#.to("cuda")
    #with open('drive/MyDrive/pcap/16.pt', 'rb') as f:
    #  model = torch.load(f, weights_only=False)
    #model.to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)

    model.train()

    size = len(dld.dataset)
    batches_num = len(dld)
    min_loss_test = None
    for epoch in range(2000):
        for batch, (data, tag) in enumerate(dld):
            model.train()
            #print(data[0].shape)
            #print(data[1].shape)
            train_loss = 0.0

            print(tag)
            tag = torch.stack(tag)
            inputs, targets = data, tag
            #print(inputs.shape)
            #input()
            # print(inputs)
            # print(targets)
            opt.zero_grad()

            preds = model(inputs)
            # print(preds)
            # input()
            loss = nn.BCEWithLogitsLoss()(preds, targets)#better_loss(preds, targets)#nn.MSELoss()(preds, targets)
            loss.backward()
            opt.step()
            train_loss += loss.item()


            if batch % 100 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print(inputs)
                # print(preds)
        loss_test = eval_model(dld_test, model)
        print("test loss", loss_test)
        if not min_loss_test or loss_test < min_loss_test:
          min_loss_test = loss_test
          with open(f'drive/{epoch}.pt', 'wb') as f:
            torch.save(model, f)
        else:
          for g in opt.param_groups:
            g['lr'] /= 4.0


if __name__ == "__main__":
    main(sys.argv)
