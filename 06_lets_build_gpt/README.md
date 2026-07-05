# Turkish Poetry GPT (Sıfırdan Türkçe Şiir Üreten GPT) 🇹🇷📝

Bu proje, Andrej Karpathy'nin ünlü "Let's build GPT" serisinden ilham alınarak **tamamen sıfırdan** PyTorch kullanılarak geliştirilmiş bir Decoder-Only Transformer (GPT) dil modelidir. Model, 19.000 Türkçe şiirden oluşan kapsamlı bir veri seti kullanılarak eğitilmiş olup, verilen bağlam üzerinden rastgele kafiyeli ve yapısal Türkçe şiirler üretebilmektedir.

## 🚀 Projenin Özeti

Bu projede devasa dil modellerinin (LLM) kalbini oluşturan **Transformer** mimarisi sıfırdan inşa edilmiştir. Kullanılan temel bileşenler şunlardır:
- **Token & Position Embeddings:** Harflerin anlamlarını ve cümledeki sıralarını matematiksel uzaya taşıyan matrisler.
- **Multi-Head Self-Attention:** Modelin bir harfe bakarken kendinden önceki tüm bağlamı (harfleri ve dil kurallarını) çok yönlü analiz ettiği "iletişim/farkındalık" katmanı.
- **FeedForward Network:** İletişim sonrası verinin işlendiği "sindirim/düşünme" katmanı.
- **Residual Connections & LayerNorm:** Derin ağların eğitimini kolaylaştıran kestirme yollar ve veri dengeleyiciler.

## ⚙️ Model Özellikleri ve Hiperparametreler

Modelin çalışması için kullanılan güncel parametreler (istediğiniz gibi `train_turkish_gpt.py` üzerinden değiştirebilirsiniz):
- `batch_size`: 64
- `block_size` (Bağlam): 128
- `n_embd` (Vektör Boyutu): 128
- `n_head` (Dikkat Kafası): 4
- `n_layer` (Blok Sayısı): 4
- `learning_rate`: 3e-4

Model cihaz olarak `MPS` (Apple Metal Performance Shaders), `CUDA` (Nvidia GPU) veya `CPU` desteklemektedir. M-serisi bir Mac kullanıyorsanız model otomatik olarak MPS'i algılar ve işlemleri ekran kartı (GPU) hızında saniyeler içinde tamamlar.

## 📚 Veri Seti (Dataset)

Eğitim için **Hugging Face** üzerinde halka açık olarak bulunan [okg/turkish-poems](https://huggingface.co/datasets/okg/turkish-poems) kullanılmıştır. 
Python betiği otomatik olarak:
1. Bu veri setini indirir.
2. Web tabanlı çöp verileri (HTML etiketlerini, `<br>`, `<p>` vb.) RegEx ile temizler.
3. Eğitime hazır devasa bir `input.txt` dosyası oluşturur.

## 💻 Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri indirin:
```bash
pip install torch pandas datasets
```

2. Eğitimi Başlatın:
```bash
python train_turkish_gpt.py
```

*Not: Veya isterseniz kodu adım adım incelemek için `turkish_gpt_demo.ipynb` dosyasını kullanabilirsiniz.*

## 🎭 Örnek Çıktılar

Modelin henüz ilk birkaç bin adımında bile öğrendiği şaşırtıcı dilbilgisi ve yapı kuralları (10-15 dakikalık eğitimle):

> *kalma ruhumdu deniz arkasında*  
> *bir ağlayan yolunu ettiğiniz temizleme*  
> *seni sabırsın yüreğini sinfir etmesine*  
> *şerefleri tazirip giden ölüm balsın*  

---
*Geliştirici:* Mikail Başer (AI Öğrenme Yol Haritası kapsamında üretilmiştir)
