# CS221 - Đồ án: INSTRUCTOR Embedding

## 📚 Giới thiệu

Đồ án này nghiên cứu và triển khai paper **"One Embedder, Any Task: Instruction-Finetuned Text Embeddings"** (INSTRUCTOR) - một mô hình text embedding có thể tạo ra embeddings tùy biến cho bất kỳ task nào chỉ bằng cách cung cấp instruction phù hợp.

### 📄 Thông tin Paper

| | |
|---|---|
| **Tên paper** | One Embedder, Any Task: Instruction-Finetuned Text Embeddings |
| **Tác giả** | Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu |
| **Tổ chức** | University of Washington, University of Hong Kong, Meta AI, Allen Institute for AI |
| **Năm** | 2022 |
| **Link** | [ACL 2023 Findings](https://aclanthology.org/2023.findings-acl.71) |

---

## 🎯 Ý tưởng chính

**INSTRUCTOR** giải quyết vấn đề: *"Làm sao để một mô hình embedding duy nhất có thể hoạt động tốt trên nhiều tasks khác nhau?"*

### Giải pháp: Instruction-Finetuned Embeddings

Thay vì train nhiều mô hình cho từng task, INSTRUCTOR sử dụng **instructions** để hướng dẫn mô hình tạo embeddings phù hợp:

```python
# Cùng một câu, khác instruction → embeddings khác nhau
["Represent the sentence for sentiment classification:", "I love this movie!"]
["Represent the sentence for topic classification:", "I love this movie!"]
["Represent the question for retrieving answers:", "I love this movie!"]
```

### Template Instruction

```
Represent the [domain] [text_type] for [task_objective]:
```

- **domain**: Lĩnh vực (science, finance, medicine, ...)
- **text_type**: Loại văn bản (sentence, document, question, ...)
- **task_objective**: Mục tiêu (classification, retrieval, clustering, ...)

---

## 🏗️ Cấu trúc dự án

```
CS221/
├── README.md                    # File này
├── instructor-embedding/        # Source code chính
│   ├── demo.ipynb              # 🎯 Notebook demo đầy đủ
│   ├── train.py                # Script huấn luyện
│   ├── requirements.txt        # Dependencies
│   ├── setup.py                # Package setup
│   │
│   ├── InstructorEmbedding/    # Core module
│   │   ├── __init__.py
│   │   └── instructor.py       # INSTRUCTOR model class
│   │
│   ├── evaluation/             # Đánh giá mô hình
│   │   ├── MTEB/              # Benchmark MTEB
│   │   ├── prompt_retrieval/   # Prompt retrieval evaluation
│   │   └── text_evaluation/    # Text evaluation metrics
│   │
│   ├── examples/               # Ví dụ sử dụng
│   │   └── faiss/             # FAISS integration
│   │
│   ├── input/                  # Training data
│   │   └── medi-data.json     # MEDI dataset
│   │
│   └── output/                 # Model outputs
```

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/AdamNbz/CS221.git
cd CS221/instructor-embedding
```

### 2. Tạo môi trường ảo với Conda

```bash
# Tạo environment mới với Python 3.9
conda create -n instructor python=3.9 -y

# Kích hoạt environment
conda activate instructor

# (Tùy chọn) Cài đặt CUDA toolkit nếu dùng GPU
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

> 💡 **Tip**: Sử dụng Python 3.8-3.10 để đảm bảo tương thích với các dependencies.

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Kiểm tra cài đặt

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')
```

---

## 💻 Hướng dẫn sử dụng

### Sử dụng cơ bản

```python
from InstructorEmbedding import INSTRUCTOR

# Load model
model = INSTRUCTOR('hkunlp/instructor-large')

# Tạo embeddings với instruction
sentences = [
    ["Represent the Science sentence:", "Quantum physics explains atomic behavior"],
    ["Represent the Finance sentence:", "Stock prices rose after the announcement"]
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 768)
```

### Tính Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Các cặp câu cần so sánh
sentences_a = [["Represent the sentence for similarity:", "The cat is sleeping"]]
sentences_b = [["Represent the sentence for similarity:", "A feline is resting"]]

emb_a = model.encode(sentences_a)
emb_b = model.encode(sentences_b)

similarity = cosine_similarity(emb_a, emb_b)
print(f"Similarity: {similarity[0][0]:.4f}")
```

### Information Retrieval

```python
import numpy as np

query = [["Represent the question for retrieving documents:", "What is machine learning?"]]
documents = [
    ["Represent the document for retrieval:", "Machine learning is a subset of AI..."],
    ["Represent the document for retrieval:", "The weather is sunny today..."],
    ["Represent the document for retrieval:", "Deep learning uses neural networks..."]
]

query_emb = model.encode(query)
doc_emb = model.encode(documents)

# Tìm document liên quan nhất
similarities = cosine_similarity(query_emb, doc_emb)
best_match = np.argmax(similarities)
print(f"Best match: Document {best_match}")
```

---

## 📊 Demo Notebook

File `demo.ipynb` chứa các demo chi tiết:

| Demo | Mô tả |
|------|-------|
| 🎯 Sentiment Analysis | Minh họa instruction kéo câu cùng sentiment gần nhau |
| 📊 Dot Product Analysis | Heatmap similarity matrix, phân tích định lượng |
| 🔍 QA Retrieval | Instruction kéo Question-Answer gần nhau |
| 📚 Domain Similarity | So sánh Science/Finance/Medical domains |
| 🔎 Document Search | Information retrieval demo |
| 🎯 Text Clustering | K-Means clustering với embeddings |
| 📈 Evaluation | STS correlation, duplicate detection |
| 🔬 Ablation Study | Ảnh hưởng của instruction khác nhau |

---

## 📈 Kết quả đánh giá

INSTRUCTOR đạt **State-of-the-Art** trên 70+ embedding tasks:

| Model | MTEB Avg. Score |
|-------|-----------------|
| instructor-base | 55.9 |
| instructor-large | 58.4 |
| **instructor-xl** | **58.8** |

### Các benchmark được hỗ trợ:

- **MTEB**: Massive Text Embedding Benchmark
- **Billboard**: Text generation evaluation
- **Prompt Retrieval**: In-context learning example selection

---

## 🔧 Huấn luyện mô hình

### Dữ liệu huấn luyện: MEDI

**M**ultitask **E**mbeddings **D**ata with **I**nstructions - 330 datasets từ:
- Super-NaturalInstructions
- Sentence-Transformers embedding data
- KILT
- MedMCQA

### Chạy huấn luyện

```bash
python train.py \
    --model_name_or_path sentence-transformers/gtr-t5-large \
    --output_dir ./output \
    --cache_dir ./input \
    --max_source_length 512 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --cl_temperature 0.1
```

---

## 📖 Tài liệu tham khảo

1. **Paper gốc**: [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://arxiv.org/abs/2212.09741)

2. **Project page**: [instructor-embedding.github.io](https://instructor-embedding.github.io/)

3. **HuggingFace Models**:
   - [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base)
   - [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)
   - [hkunlp/instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

4. **GitHub chính thức**: [HKUNLP/instructor-embedding](https://github.com/HKUNLP/instructor-embedding)

---

## 📝 Citation

```bibtex
@inproceedings{INSTRUCTOR,
  title={One Embedder, Any Task: Instruction-Finetuned Text Embeddings},
  author={Su, Hongjin and Shi, Weijia and Kasai, Jungo and Wang, Yizhong and Hu, Yushi and Ostendorf, Mari and Yih, Wen-tau and Smith, Noah A. and Zettlemoyer, Luke and Yu, Tao},
  url={https://arxiv.org/abs/2212.09741},
  year={2022},
}
```

---

## 👥 Thông tin đồ án

| | |
|---|---|
| **Môn học** | CS221 |
| **Repository** | [github.com/AdamNbz/CS221](https://github.com/AdamNbz/CS221) |

---

<p align="center">
  <b>🎓 CS221 - INSTRUCTOR Embedding Project</b>
</p>
