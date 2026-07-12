import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

# -----------------------------------------------------------------------------
# HYPERPARAMETERS / HİPERPARAMETRELER
# These values control the size and training behavior of our model.
# Bu değerler modelimizin boyutunu ve eğitim davranışını kontrol eder.
# -----------------------------------------------------------------------------
batch_size = 64         # Number of independent sequences processed in parallel (Aynı anda işlenecek bağımsız metin parçası sayısı)
block_size = 128        # Maximum context length for predictions (Modelin aynı anda bakabildiği maksimum geçmiş harf sayısı - Bağlam penceresi)
max_iters = 5000        # Total number of training steps (Toplam eğitim döngüsü sayısı)
eval_interval = 500     # How often to evaluate the loss (Kaç adımda bir test hatasını hesaplayacağımız)
learning_rate = 3e-4    # Step size for the optimizer (Öğrenme hızı: Modelin her adımda kendini ne kadar güncelleyeceği)
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # Hardware accelerator (Donanım hızlandırıcı: Apple Silicon için MPS)
eval_iters = 200        # Number of batches to average during evaluation (Test hatasını hesaplarken ortalaması alınacak örneklem sayısı)
n_embd = 128            # Embedding dimension / The "brain capacity" of a token (Karakterlerin matematiksel uzaydaki boyutu/derinliği)
n_head = 4              # Number of attention heads (Aynı anda çalışan "detektif" / dikkat kafası sayısı)
n_layer = 4             # Number of Transformer blocks (Üst üste dizilecek Transformer bloklarının sayısı - Ağın derinliği)
dropout = 0.2           # Probability of dropping a neuron to prevent overfitting (Aşırı öğrenmeyi/ezberlemeyi önlemek için nöron uyutma/kapatma oranı)
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# 1. DATA PREPARATION / VERİ HAZIRLIĞI
# Loading, cleaning, and preparing the Turkish Poetry dataset.
# Türkçe Şiir veri setinin yüklenmesi, temizlenmesi ve hazırlanması.
# -----------------------------------------------------------------------------
if not os.path.exists("input.txt"):
    print("Downloading dataset... / Veri seti indiriliyor...")
    url = "https://huggingface.co/datasets/okg/turkish-poems/resolve/main/poems.csv"
    df = pd.read_csv(url)
    
    # Extract the 'poem' column and drop empty ones (Şiir metinlerini al ve boş olanları at)
    poems = df['poem'].dropna().astype(str).tolist()
    
    clean_text = ""
    for poem in poems:
        # Clean HTML tags and line breaks (Satır atlamalarını düzelt ve HTML etiketlerini regex ile tamamen temizle)
        poem = poem.replace('<br>', '\n').replace('\r', '')
        poem = re.sub(r'<[^>]+>', '\n', poem)
        clean_text += poem + "\n\n"
        
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)

# Read the entire dataset into a single string (Tüm veri setini tek bir devasa metin olarak oku)
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Find all unique characters in the text (Metindeki tüm benzersiz harfleri/sembolleri bul)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (Harfleri sayılara, sayıları harflere dönüştüren sözlükleri oluştur)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]     # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers -> string

# Convert text to a PyTorch tensor (Tüm metni PyTorch tensörüne çevir)
data = torch.tensor(encode(text), dtype=torch.long)

# Train/Validation split: 90% for training, 10% for validation (Verinin %90'ı eğitim, %10'u test için ayrılır)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # Generate a small batch of data of inputs (x) and targets (y)
    # Girdi (x) ve hedef (y) verilerinden oluşan rastgele küçük paketler (batch) üretir
    data_split = train_data if split == 'train' else val_data
    # Pick random starting indexes (Rastgele başlangıç noktaları seç)
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    #433Geçmiş harfleri üst üste diz)
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    # Stack the target characters to predict (Tahmin edilecek gelecek harfleri üst üste diz)
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    # Move the data to the device (MPS/CPU) (Veriyi ekran kartına veya işlemciye taşı)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Tell PyTorch not to calculate gradients here (Burada gradyan/türev hesaplama ki RAM şişmesin)
def estimate_loss():
    # Evaluate the model's loss on both train and validation sets
    # Modelin hem eğitim hem de test verisi üzerindeki hatasını ortalama alarak hesaplar
    out = {}
    m.eval() # Switch model to evaluation mode (Modeli test moduna al - dropout kapanır)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train() # Switch back to training mode (Eğitim moduna geri dön)
    return out


# -----------------------------------------------------------------------------
# 2. TRANSFORMER ARCHITECTURE / MİMARİ BİLEŞENLER
# Building the GPT from the ground up.
# GPT mimarisinin temelden inşa edilmesi.
# -----------------------------------------------------------------------------

class Head(nn.Module):
    """ One head of self-attention (Tek bir Self-Attention / Öz-Dikkat kafası) """

    def __init__(self, head_size):
        super().__init__()
        # Key: "What do I contain?" (Ben neyim? - Benim özelliğim ne?)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # Query: "What am I looking for?" (Ne arıyorum? - Kendime uygun ne bulabilirim?)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # Value: "If you find me interesting, here is my actual information" (Beni ilginç bulursan asıl bilgim budur)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril: The lower triangular matrix for masking (Geleceği gizleyen o meşhur üçgen matris)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute attention scores ("Affinities") - Dikkat skorlarının (yakınlık/ilgi) hesaplanması
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # Mask out the future tokens (Gelecekteki harflerin üstünü eksi sonsuz ile ört, sadece geçmişe baksın)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax: convert to percentages (Sonsuzlukları ve skorları 0 ile 1 arasında yüzdeliklere dönüştür)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # Prevent over-reliance on specific tokens (Ezberlemeyi engellemek için bazılarının gözünü rastgele kapat)
        
        # Aggregate the values based on attention scores (Yüzdeliklere göre asıl bilgileri / Value / topla)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel (Aynı anda çalışan birkaç farklı dikkat kafası) """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a list of parallel heads (Paralel çalışacak kafaları oluştur)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer: blend the outputs (Kafalardan çıkan farklı sonuçları harmanlayan katman)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate all head outputs (Tüm kafaların sonuçlarını yan yana birleştir)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Pass through projection and dropout (Harmanla ve çıkışa gönder)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity (Bireysel düşünme / Sindirim odası) """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Multiply dimensions by 4 for "thinking" space (Düşünmek için boyutu 4 katına çıkar)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # Non-linear activation (Doğrusal olmayan işlem: Zeka burada oluşur)
            # Project back to original dimensions (Tekrar orijinal boyuta geri dön)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation (Transformer Bloğu: Önce iletişim, sonra bireysel düşünme) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # Step 1: Communication phase (İletişim/Dikkat aşaması - Harfler birbirleriyle konuşur)
        self.sa = MultiHeadAttention(n_head, head_size)
        # Step 2: Computation phase (Hesaplama/Düşünme aşaması - Harfler öğrendiklerini sindirir)
        self.ffwd = FeedForward(n_embd)
        # Normalization layers to keep gradients stable (Patlamaları ve kaybolmaları önleyen dengeleyici katmanlar)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x = x + ... Residual Connections! (Kestirme yollar! Gradyanların ağın en derinine inebilmesini sağlar)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token Embedding: Map each character to a vector (Her bir harfi matematiksel bir vektöre dönüştür - Harfin Kimliği)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position Embedding: Map each position to a vector (Harfin cümle içindeki sırasını/konumunu belirten vektör)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack multiple Transformer blocks (Ağı derinleştirmek için Transformer bloklarını üst üste diz)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final layer normalization (Çıkıştan önceki son dengeleme)
        self.ln_f = nn.LayerNorm(n_embd)
        # Language Modeling Head: Predict the next character (Son tahmin katmanı: Bir sonraki harf ne olacak?)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Fetch embeddings (Harfin anlamını ve pozisyonunu tablodan çek)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        
        # 2. Add them together (Harfin kimliği ile cümledeki yerini toplayarak birleştir)
        x = tok_emb + pos_emb # (B,T,C)
        
        # 3. Pass through the Transformer blocks (Veriyi o devasa "beyin" bloklarından geçir)
        x = self.blocks(x)    # (B,T,C)
        x = self.ln_f(x)      # (B,T,C)
        
        # 4. Predict the logits/scores for the next character (Sıradaki harf için skorları hesapla)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # If we have targets, calculate the Cross Entropy Loss (Hedefler belliyse tahmin hatamızı hesapla)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Function to generate new text token by token (Harf harf yeni şiir üretme fonksiyonu)
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (Eğer üretilen metin çok uzadıysa, sadece son 128 harfe odaklan)
            idx_cond = idx[:, -block_size:]
            
            # Get predictions (Tahmin skorlarını al)
            logits, loss = self(idx_cond)
            # Focus only on the very last time step (Sadece en sondaki harfin ürettiği tahmine odaklan)
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities (Skorları yüzdelik ihtimallere çevir)
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution (En yüksek ihtimale göre rastgele bir harf seç)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence (Seçilen yeni harfi metnin sonuna ekle ve döngüye devam et)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# 3. TRAINING LOOP / EĞİTİM DÖNGÜSÜ
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize the model and move it to the device (MPS/CPU) (Modeli yarat ve ekran kartına taşı)
    m = GPTLanguageModel()
    m = m.to(device)
    
    # Create a PyTorch optimizer (AdamW is the standard for Transformers) (Ağırlıkları güncelleyecek optimizasyon motorunu kur)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        # Periodically evaluate the loss on train and val sets (Belirli aralıklarla modelin ne kadar iyi öğrendiğini test et)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # Get a batch of data (Yeni bir veri paketi çek)
        xb, yb = get_batch('train')
        
        # Forward pass: predict targets and calculate loss (İleri besleme: Tahmin yap ve hatayı bul)
        logits, loss = m(xb, yb)
        
        # Backward pass: calculate gradients (Geri yayılım: Hatayı nöronlara dağıtarak kimin ne kadar suçlu olduğunu bul)
        optimizer.zero_grad(set_to_none=True) # Reset old gradients (Eski hataları sıfırla)
        loss.backward()
        
        # Update weights (Ağırlıkları güncelle ve modeli biraz daha akıllı hale getir)
        optimizer.step()
    
    # -----------------------------------------------------------------------------
    # 4. GENERATION / ÜRETİM
    # -----------------------------------------------------------------------------
    # Kick off generation with a single zero token (newline) (Üretime sadece bir "Enter" boşluğu vererek başla)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Generate 500 characters and decode them back to text (500 yeni harf üret ve sayılardan metne çevirip ekrana yaz)
    print("\n--- MODELİN ÜRETTİĞİ ŞİİR / GENERATED POEM ---\n")
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
