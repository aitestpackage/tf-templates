"""
!!pip install muon_optimizer
!pip install sacrebleu
"""
import random
f = open("/kaggle/input/frenchenglish-bilingual-pairs/fra.txt", "r")
r = f.read().splitlines()
f.close()
random.shuffle(r)
els = []
frs = []
for y in r:
    el, fr = y.split("\t")
    els.append(el)
    frs.append(fr)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
import os
def maketokenizer(sents, of):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(
        vocab_size=2048,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    tokenizer.train_from_iterator(sents, trainer=trainer)
    if os.path.exists(of):
        os.remove(of)
    tokenizer.save(of)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=of,
                                        unk_token="[UNK]",
                                        pad_token="[PAD]",
                                        bos_token="[BOS]",
                                        eos_token="[EOS]", padding_side="left")
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A $B [EOS]",
        special_tokens=[("[BOS]", tokenizer.bos_token_id), ("[EOS]", tokenizer.eos_token_id)]
    )
    return tokenizer

el_tokenizer = maketokenizer(els, "el_tokenizer.json")
fr_tokenizer = maketokenizer(frs, "fr_tokenizer.json")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

el_t = el_tokenizer(els, return_tensors="pt", max_length = 32, padding="max_length", truncation=True)["input_ids"]
fr_t = fr_tokenizer(frs, return_tensors="pt", max_length = 33, padding="max_length", truncation=True)["input_ids"]

from sklearn.model_selection import train_test_split
el_train, el_test, fr_train, fr_test = train_test_split(el_t, fr_t, test_size=0.025)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
device

from torch.utils.data import Dataset, DataLoader
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, el, fr):
        super().__init__()
        self.el = np.array(el)
        self.fr = np.array(fr)
    def __len__(self):
        return self.el.shape[0]
    def __getitem__(self, idx):
        return self.el[idx], self.fr[idx]

train_ds = TranslationDataset(el_train, fr_train)
test_ds = TranslationDataset(el_test, fr_test)


def collate_fn(batch):
    el = []
    fr = []
    for el_t, fr_t in batch:
        el.append(el_t)
        fr.append(fr_t)
    el = np.array(el)
    fr = np.array(fr)
    return torch.from_numpy(el), torch.from_numpy(fr)


train_loader = DataLoader(
    train_ds,
    batch_size=512,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

test_loader = DataLoader(
    test_ds,
    batch_size=1024,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

for el, fr in train_loader:
    print(el.shape)
    print(fr.shape)
    break

def relu2(x):
    x = F.relu(x)
    # x = F.gelu(x)
    x = torch.square(x)
    return x

class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.relu(x)
        x = torch.square(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            ReLU2(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.alpha = nn.Parameter(torch.tensor([1.0, 1.0]) * 1.0)

    def forward(self, x, mask):
        x1 = self.norm1(x)
        # x1 = x
        x1, _ = self.self_attn(x1, x1, x1, attn_mask=mask)
        x = x + x1 * self.alpha[0]
        x1 = self.norm2(x)
        x1 = self.ffn(x1)
        return x + x1 * self.alpha[1]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            ReLU2(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.alpha = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]) * 1.0)

    def forward(self, x, memory, tgt_mask):
        x1 = self.norm1(x)
        x1, _ = self.self_attn(x1, x1, x1, attn_mask=tgt_mask)
        x1 = self.dropout1(x1)
        x = x + x1 * self.alpha[0]
        x1 = self.norm2(x)
        x1, _ = self.cross_attn(x1, memory, memory, attn_mask=None) # No need to mask encoder memory in NMT
        x1 = self.dropout2(x1)
        x = x + x1 * self.alpha[1]
        x = self.norm3(x)
        x = x + self.ffn(x) * self.alpha[2]
        return x

import torch.nn as nn
import torch, math
from einops import rearrange
import torch.nn.functional as F

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()

        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU()
        ))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU()
            ))
        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device):
        indices = (n-1) + torch.arange(n).unsqueeze(1) - torch.arange(n).unsqueeze(0)
        pos = torch.arange(-n + 1, n, device = device).float().unsqueeze(-1)
        for layer in self.mlp:
            pos = layer(pos)
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

# self.pe = DynamicPositionBias(embed_dim // 4, n_heads, 3)
# ...
# logits = F.scaled_dot_product_attention(
#     q, k, v,
#     attn_mask=F.pad(self.pe(sl, x.device), (0, 1)),
#     is_causal=False,
#     dropout_p=0.0
# )

class EncoderDecoderwAlibi(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, in_vocab_size, out_vocab_size, seq_len, num_layers):
        super().__init__()
        self.src_embedding = nn.Embedding(in_vocab_size, hidden_dim)
        self.trg_embedding = nn.Embedding(out_vocab_size, hidden_dim)

        self.encoder = nn.ModuleList([EncoderLayer(hidden_dim, 8, ffn_dim) for _ in range(num_layers[0])])
        self.decoder = nn.ModuleList([DecoderLayer(hidden_dim, 8, ffn_dim) for _ in range(num_layers[1])])

        self.alibi_m = [1 / (2 ** i) for i in range(1, 9)]
        x = torch.arange(seq_len)
        y = torch.arange(seq_len).unsqueeze(-1)
        self.alibi_val = x - y
        self.alibi_val = self.alibi_val.to(device).unsqueeze(0)
        self.alibi_val.requires_grad = False

        self.pe = DynamicPositionBias(hidden_dim // 4, 8, 3)
        # self.pe.require_grad = False

        self.causal_mask = torch.ones(1, seq_len, seq_len, requires_grad=False, device=device) * (float('-inf'))
        self.causal_mask = torch.triu(self.causal_mask, diagonal=1)

        self.output = nn.Linear(hidden_dim, out_vocab_size)

    def forward(self, src, trg):
        batch_size = src.shape[0]
        x = self.src_embedding(src)
        # x = F.rms_norm(x, (x.size(-1),))

        # MASK COMPUTATION
        # alibi_mask = (self.alibi_val * self.alibi_m[0]).expand(batch_size, -1, -1)
        # for i in range(1, 8):
        #     alibi_mask = torch.cat([alibi_mask, (self.alibi_val * self.alibi_m[i]).expand(batch_size, -1, -1)])
        # alibi_mask = torch.tril(alibi_mask)
        alibi_mask = self.pe(x.shape[1], device=device)
        alibi_mask = alibi_mask.repeat(batch_size, 1, 1)
        # END MASK COMPUTATION

        # x = self.encoder(x, mask=alibi_mask)
        for layer in self.encoder:
            x = layer(x, mask=alibi_mask)
        # x = F.rms_norm(x, (x.size(-1),))

        # DECODER MASK
        mask = self.causal_mask.expand(batch_size * 8, -1, -1)
        mask = mask + alibi_mask
        # END OF DECODER MASK

        trg = self.trg_embedding(trg)
        trg_len = trg.shape[1]
        mask = mask[:, :trg_len, :trg_len]
        # x = self.decoder(
        #     tgt=trg,
        #     memory=x,
        #     tgt_mask=mask
        # )
        # return self.output(x)
        for layer in self.decoder:
            trg = layer(trg, x, mask)
        return self.output(trg)


model = EncoderDecoderwAlibi(
    hidden_dim=32,
    ffn_dim=16,
    in_vocab_size=2048,
    out_vocab_size=2048,
    seq_len=32,
    num_layers=(1, 1),
).to(device)

for el, fr in train_loader:
    print(el.shape)
    print(fr.shape)
    output = model(el.to(device), fr[:, :-1].to(device))
    print(output.shape)
    # print(output[0])
    break

from tqdm import tqdm

def train():
    model.train()
    total_loss = 0
    cnt = 0
    prev_avg = 999999.9
    for batch in (pbar := tqdm(train_loader)):
        el, fr = batch
        el, fr = el.to(device), fr.to(device)
        fr_in = fr[:, :-1]
        fr_actual = fr[:, 1:]
        pred = model(el, fr_in)
        loss = criteria(pred.reshape(-1, 2048), fr_actual.reshape(-1))
        total_loss += loss.item()
        cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Average Loss: {total_loss / cnt :6f}")
        if cnt % 20 == 0:
            # if total_loss / cnt < prev_avg:
            #     prev_avg = total_loss / cnt
            # else:
            #     if total_loss / cnt > prev_avg * 1.02:
            #         scheduler.step()
            #         print("Stepping Scheduler")
            cnt = 0
            total_loss = 0
    # if scheduler is not None:
    #     scheduler.step()


def test():
    model.eval()
    total_loss = 0
    cnt = 0
    num_correct = 0
    total_cnt = 0
    with torch.no_grad():
        for batch in (pbar := tqdm(test_loader)):
            el, fr = batch
            el, fr = el.to(device), fr.to(device)
            fr_in = fr[:, :-1]
            fr_actual = fr[:, 1:]
            pred = model(el, fr_in)
            loss = criteria(pred.reshape(-1, 2048), fr_actual.reshape(-1))
            total_loss += loss.item()
            cnt += 1
            # Obtain the token prediction
            pred_argmax = pred.argmax(-1)
            pred_argmax = pred_argmax.reshape(-1)
            fr_actual = fr_actual.reshape(-1)
            # Mask out pad token to not count towards final accuracy
            not_pad_mask = (fr_actual != 0)
            pred_argmax = pred_argmax[not_pad_mask]
            fr_actual = fr_actual[not_pad_mask]
            # Compute accuracy
            num_correct += torch.sum(fr_actual == pred_argmax).cpu().numpy()
            total_cnt += fr_actual.shape[0]
            pbar.set_description(f"Testing Loss: {total_loss / cnt :6f} | Testing Accuracy: {num_correct / total_cnt : 6f}")


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class MultipleScheduler(object):
    def __init__(self, *op):
        self.schedulers = op

    def step(self):
        for op in self.schedulers:
            op.step()


model = EncoderDecoderwAlibi(
    hidden_dim=128,
    ffn_dim=280,
    in_vocab_size=2048,
    out_vocab_size=2048,
    seq_len=32,
    num_layers=(3, 5),
).to(device)


from torch.optim import *
from torch.optim.lr_scheduler import *

from muon import SingleDeviceMuon
hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]

param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=1.5e-2, weight_decay=0, momentum=0.95, nesterov=True),
]
muon_optimizer = SingleDeviceMuon(param_groups)
muon_scheduler = ExponentialLR(muon_optimizer, gamma=0.6)

param_groups = [
    dict(params=hidden_gains_biases, lr=5e-3, weight_decay=0),
]
adamw_optimizer = AdamW(param_groups)
adamw_scheduler = ExponentialLR(adamw_optimizer, gamma=0.8)

criteria = nn.CrossEntropyLoss(ignore_index=0)
# class_weights = torch.ones(2048).to(device)
# class_weights[0] = 0.0
# class_weights[fr_tokenizer.eos_token_id] = 100.0
# criteria = nn.CrossEntropyLoss(weight = class_weights)
optimizer = MultipleOptimizer(muon_optimizer, adamw_optimizer)
scheduler = MultipleScheduler(muon_scheduler, adamw_scheduler)

for _ in range(3):
    train()
    test()

def translate(input_str):
    model.eval()
    en_ids = el_tokenizer(input_str, return_tensors="pt", max_length = 32, padding="max_length", truncation=True)["input_ids"]
    en_ids = en_ids.to(device)
    # print(en_ids)
    fr_ids = [2] # The first token
    with torch.no_grad():
        while len(fr_ids) <= 32:
            # print(torch.tensor(fr_ids).to(torch.long).to(device).unsqueeze(0))
            pred = model(en_ids, torch.tensor(fr_ids).to(torch.long).to(device).unsqueeze(0))
            pred = pred[0].argmax(-1)
            fr_ids.append(int(pred[-1].cpu().numpy()))
            # print(fr_ids)
    return fr_ids


def translate_for_bleu(en_str, max_len=32):
    model.eval()
    en_ids = el_tokenizer(en_str, return_tensors="pt", max_length = 32, padding="max_length", truncation=True)["input_ids"]
    en_ids = en_ids.to(device)
    # print(en_ids)
    fr_ids = [2] # The first token
    with torch.no_grad():
        while len(fr_ids) <= max_len:
            # print(torch.tensor(fr_ids).to(torch.long).to(device).unsqueeze(0))
            pred = model(en_ids, torch.tensor(fr_ids).to(torch.long).to(device).unsqueeze(0))
            pred = pred[0].argmax(-1)
            fr_ids.append(int(pred[-1].cpu().numpy()))
            # print(fr_ids)
            if fr_ids[-1] == fr_tokenizer.eos_token_id:
                break
    return fr_ids[1:-1]


import sacrebleu

hypotheses = []
references = []
model.eval()
bleu_loader = DataLoader(
    train_ds,
    batch_size=1024,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)
with torch.no_grad():
    batch = next(iter(bleu_loader))
    src_batch, fr_batch = batch
    fr_in_batch, fr_t_batch = fr_batch[:, :-1], fr_batch[:, 1:]
    for ref_ids in fr_t_batch:
        ref_str = fr_tokenizer.decode(ref_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
        references.append(ref_str)

    for src_ids in tqdm(src_batch):
        src_str = el_tokenizer.decode(src_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
        hyp_str = translate_for_bleu(src_str, len(src_ids))
        pred_string = fr_tokenizer.decode(hyp_str)
        hypotheses.append(pred_string)

model.train()
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

print(f"Corpus BLEU = {bleu.score:.2f}")


