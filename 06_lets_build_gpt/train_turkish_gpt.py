import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Hyperparameters (Hiperparametreler)
# -----------------------------------------------------------------------------
batch_size = 64         # Aynı anda işlenecek bağımsız metin parçası sayısı
block_size = 128        # Modelin aynı anda bakabildiği maksimum geçmiş harf sayısı
max_iters = 5000        # Eğitim döngüsü sayısı
eval_interval = 500     # Kaç adımda bir test hatasını hesaplayalım
learning_rate = 3e-4    # Öğrenme hızı (Ne kadar büyük adım atacak)
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # Apple Silicon (M-serisi) veya CPU
eval_iters = 200        # Test hatasını hesaplarken kaç kere örneklem alalım
n_embd = 128            # Harflerin matematiksel derinliği (Özellik sayısı)
n_head = 4              # Multi-Head Attention'daki kafa (detektif) sayısı
n_layer = 4             # Üst üste dizilecek Block sayısı (Ağın derinliği)
dropout = 0.2           # Aşırı öğrenmeyi (ezberlemeyi) önlemek için nöron kapatma oranı
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# 1. Veri Hazırlığı ve Yükleme
# -----------------------------------------------------------------------------
print("Veri hazırlığı yapılıyor...")
if not os.path.exists("input.txt"):
    print("Şiirler HuggingFace'ten indiriliyor...")
    url = "https://huggingface.co/datasets/okg/turkish-poems/resolve/main/poems.csv"
    df = pd.read_csv(url)
    
    # 'poem' sütunundaki şiir metinlerini al
    siirler = df['poem'].dropna().astype(str).tolist()
    
    temiz_metin = ""
    for siir in siirler:
        # Satır atlamaları düzelt ve HTML etiketlerini (örn: <br/>, <p>) tamamen temizle
        siir = siir.replace('<br>', '\n').replace('\r', '')
        siir = re.sub(r'<[^>]+>', '\n', siir) 
        temiz_metin += siir + "\n\n"
        
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(temiz_metin)
    print("input.txt başarıyla oluşturuldu ve HTML etiketlerinden arındırıldı!")

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Eşsiz karakterleri bul ve sözlük oluştur
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Veriyi Torch Tensor'üne çevir ve Train/Validation olarak ayır
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out

# -----------------------------------------------------------------------------
# 2. GPT Modeli Mimari Tanımlamaları
# -----------------------------------------------------------------------------

class Head(nn.Module):
    """ Tek bir Self-Attention Kafası """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Dikkat (Attention) skorlarını hesapla
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # Geleceği görmesini engelle (Sadece geçmişe bakabilir)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # Ezberlemeyi önle
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Birkaç Attention kafasının paralel çalışması """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Bireysel düşünme / Sindirim odası """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Bloğu: İletişim (Attention) + Düşünme (FeedForward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x = x + ... Residual Connection (Kestirme yollar)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # Harfin anlamı + Konumu
        x = self.blocks(x)    # Bloklardan geç
        x = self.ln_f(x)      # Son dengeleme
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# 3. Modelin Eğitimi
# -----------------------------------------------------------------------------
print("Model ayağa kaldırılıyor...")
m = GPTLanguageModel()
m = m.to(device)

print(f"Eğitim başlıyor... Kullanılan Cihaz: {device}")
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Belirli aralıklarla Loss durumunu ekrana yazdır
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Adım {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# 4. Metin Üretimi (Eğitim Sonrası Test)
# -----------------------------------------------------------------------------
print("\n--- Modelden Örnek Şiir Çıktısı ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
