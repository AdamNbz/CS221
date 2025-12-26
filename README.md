# CS221 - Đồ án: INSTRUCTOR Embedding

<p align="center">
  <img src="instructor-embedding/instructor.png" alt="INSTRUCTOR Architecture" width="600"/>
</p>

## 📚 Giới thiệu

Đồ án này nghiên cứu và triển khai paper **"One Embedder, Any Task: Instruction-Finetuned Text Embeddings"** (INSTRUCTOR) - một mô hình text embedding có thể tạo ra embeddings tùy biến cho bất kỳ task nào chỉ bằng cách cung cấp instruction phù hợp.

### 📄 Thông tin Paper

| | |
|---|---|
| **Tên paper** | One Embedder, Any Task: Instruction-Finetuned Text Embeddings |
| **Tác giả** | Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu |
| **Tổ chức** | University of Washington, University of Hong Kong, Meta AI, Allen Institute for AI |
| **Năm** | 2022 |
| **Link** | [arXiv](https://arxiv.org/abs/2212.09741) / [ACL 2023 Findings](https://aclanthology.org/2023.findings-acl.71) |

---

## 🎯 Ý tưởng chính

**INSTRUCTOR** giải quyết vấn đề: *"Làm sao để một mô hình embedding duy nhất có thể hoạt động tốt trên nhiều tasks khác nhau?"*

### Giải pháp: Instruction-Finetuned Embeddings

Thay vì train nhiều mô hình cho từng task, INSTRUCTOR sử dụng **instructions** để mô tả **mục đích của task** mà model cần thực hiện:

```python
# Instruction mô tả MỤC ĐÍCH TASK, không phải bổ sung ý nghĩa cho text
["Represent the question for retrieving documents:", "What is machine learning?"]
["Represent the document for retrieval:", "Machine learning is a subset of AI..."]
["Represent the sentence for classification:", "This movie is great!"]
```

### Template Instruction

```
Represent the [domain] [text_type] for [task_objective]:
```

- **text_type**: Loại văn bản (sentence, document, question, query...)
- **task_objective**: Mục tiêu task (classification, retrieval, clustering...)
- **domain** (tùy chọn): Lĩnh vực (science, finance, news...)

> ⚠️ **Lưu ý quan trọng**: Instruction KHÔNG phải để bổ sung ngữ nghĩa cho text (ví dụ: "Apple là công ty" vs "Apple là trái cây"). Instruction là để **mô tả task** mà model cần thực hiện, giúp model biết cách tạo embedding phù hợp cho task đó.

---

## 🏗️ Cấu trúc dự án

```
CS221/
├── README.md                   # File này
├── instructor-embedding/       # Source code chính
│   ├── demo.ipynb              # Notebook demo: GTR-T5 vs INSTRUCTOR
│   ├── train.py                # Script huấn luyện
│   ├── requirements.txt        # Dependencies
│   ├── setup.py                # Package setup
│   ├── instructor.png          # Hình minh họa kiến trúc
│   │
│   ├── InstructorEmbedding/    # Core module
│   │   ├── __init__.py
│   │   └── instructor.py       # INSTRUCTOR model class
│   │
│   ├── input/                  # Training data
│   │   └── medi-data.json      # MEDI dataset (1.4M+ samples)
│   │
│   └── evaluation/             # Evaluation tools
│       ├── MTEB/               # MTEB benchmark
│       ├── prompt_retrieval/   # Prompt retrieval evaluation
│       └── text_evaluation/    # Text evaluation
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
from sentence_transformers import SentenceTransformer

# Load INSTRUCTOR pretrained
model = INSTRUCTOR('hkunlp/instructor-large')

# Load GTR-T5 backbone (để so sánh)
gtr_model = SentenceTransformer('sentence-transformers/gtr-t5-large')
```

---

## 💻 Hướng dẫn sử dụng

### So sánh GTR-T5 vs INSTRUCTOR

```python
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models
gtr_model = SentenceTransformer('sentence-transformers/gtr-t5-large')  # Backbone
instructor_model = INSTRUCTOR('hkunlp/instructor-large')  # Pretrained with instruction

# Dữ liệu mẫu
query = "How do neural networks learn?"
documents = [
    "Neural networks are inspired by the human brain.",
    "The Eiffel Tower is located in Paris, France.",
]

# GTR-T5: Encode text thuần (không instruction)
query_emb_gtr = gtr_model.encode([query])
doc_embs_gtr = gtr_model.encode(documents)

# INSTRUCTOR: Encode với instruction mô tả task
query_instruction = "Represent the question for retrieving supporting documents:"
doc_instruction = "Represent the document for retrieval:"

query_emb_inst = instructor_model.encode([[query_instruction, query]])
doc_embs_inst = instructor_model.encode([[doc_instruction, doc] for doc in documents])

# So sánh similarity
print("GTR-T5:", cosine_similarity(query_emb_gtr, doc_embs_gtr))
print("INSTRUCTOR:", cosine_similarity(query_emb_inst, doc_embs_inst))
```

### Task-Specific Instructions (chỉ INSTRUCTOR có khả năng này)

```python
text = "Machine learning is transforming how we analyze data"

# GTR-T5: Chỉ tạo 1 embedding duy nhất
emb_gtr = gtr_model.encode([text])

# INSTRUCTOR: Cùng text, instruction khác → embedding khác
task_instructions = {
    "retrieval": "Represent the question for retrieving relevant documents:",
    "classification": "Represent the sentence for classification:",
    "clustering": "Represent the sentence for clustering:",
}

for task, instruction in task_instructions.items():
    emb = instructor_model.encode([[instruction, text]])
    print(f"{task}: shape={emb.shape}")
```

### Information Retrieval với INSTRUCTOR

```python
import numpy as np

query = [["Represent the question for retrieving documents:", "What is machine learning?"]]
documents = [
    ["Represent the document for retrieval:", "Machine learning is a subset of AI..."],
    ["Represent the document for retrieval:", "The weather is sunny today..."],
    ["Represent the document for retrieval:", "Deep learning uses neural networks..."]
]

query_emb = instructor_model.encode(query)
doc_emb = instructor_model.encode(documents)

# Tìm document liên quan nhất
similarities = cosine_similarity(query_emb, doc_emb)
best_match = np.argmax(similarities)
print(f"Best match: Document {best_match}")
```

---

## 📊 Demo Notebook

File `demo.ipynb` so sánh **GTR-T5 (backbone, không instruction)** với **INSTRUCTOR (pretrained, có instruction)**:

### Phần 1: Demo với dữ liệu mẫu

| Demo | Mô tả |
|------|-------|
| 📦 **Load Models** | Load GTR-T5 và INSTRUCTOR pretrained |
| 🔍 **Demo 1: Document Retrieval** | So sánh GTR-T5 vs INSTRUCTOR trên task retrieval với heatmap visualization |
| 🎯 **Demo 2: Task-Specific Instructions** | Cùng text, instruction cho task khác nhau → embedding khác nhau (chỉ INSTRUCTOR có khả năng này) |
| 📚 **Demo 3: Retrieval Performance** | Test retrieval trên corpus lớn hơn |
| 📊 **Demo 4: Clustering** | So sánh clustering với metrics (ARI, Silhouette) và t-SNE visualization |

### Phần 2: Demo với dữ liệu MEDI thực tế

| Demo | Mô tả |
|------|-------|
| 📁 **Load MEDI Data** | Load 50,000 samples từ bộ dữ liệu MEDI-data.json |
| ⚡ **Triplet Comparison** | So sánh margin (sim_pos - sim_neg) giữa GTR-T5 và INSTRUCTOR |
| 📈 **Retrieval Metrics** | Đánh giá Recall@K, MRR trên nhiều task types |
| 📊 **Per-Task Analysis** | Phân tích chi tiết hiệu quả theo từng loại task |

### Kết luận từ Demo:

| Model | Approach | Kết quả |
|-------|----------|---------|
| **GTR-T5** | Encode text thuần, không hiểu instruction | Baseline performance |
| **INSTRUCTOR** | Encode [instruction, text], hiểu task context | Accuracy, Margin, Recall đều cao hơn |

> **One Embedder, Any Task**: INSTRUCTOR vượt trội GTR-T5 nhờ instruction giúp model tạo embedding phù hợp với từng task cụ thể.

---

## 📈 Kết quả đánh giá

### Benchmark: MTEB (Massive Text Embedding Benchmark)

INSTRUCTOR đạt **State-of-the-Art** trên 70+ embedding tasks:

| Model | MTEB Avg. Score | Parameters |
|-------|-----------------|------------|
| instructor-base | 55.9 | 110M |
| instructor-large | 58.4 | 335M |
| **instructor-xl** | **58.8** | 1.5B |

### Kết quả từ Demo Notebook (GTR-T5 vs INSTRUCTOR)

| Metric | GTR-T5 (Backbone) | INSTRUCTOR (Pretrained) | Improvement |
|--------|-------------------|-------------------------|-------------|
| Triplet Accuracy | ~70-80% | ~85-95% | +10-15% |
| Average Margin | ~0.05 | ~0.15 | +0.10 |
| Clustering Silhouette | Lower | Higher | Varies |
| Retrieval MRR | Baseline | Higher | Varies by task |

> Kết quả cụ thể tùy thuộc vào dữ liệu test. Xem chi tiết trong `demo.ipynb`.

---

## 🔧 Huấn luyện mô hình

### Dữ liệu huấn luyện: MEDI

**M**ultitask **E**mbeddings **D**ata with **I**nstructions:
- **1,435,000 training examples** từ 330 datasets
- Mỗi sample gồm: `query`, `pos` (positive), `neg` (negative)
- Format: `[instruction, text]` cho mỗi phần
- Sources: Super-NaturalInstructions, Sentence-Transformers, KILT, MedMCQA

### Chạy huấn luyện

```bash
cd instructor-embedding

python train.py \
    --model_name_or_path sentence-transformers/gtr-t5-large \
    --output_dir ./output \
    --cache_dir ./input \
    --max_source_length 512 \
    --num_train_epochs 10 \
    --save_steps 500 \
    --cl_temperature 0.1 \
    --warmup_ratio 0.1 \
    --learning_rate 2e-5 \
    --overwrite_output_dir \
    --max_examples {n}
```

> 💡 **Lưu ý**: `--max_examples {n}` giới hạn số lượng training samples để giảm thời gian training, với n là số lượng samples.

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
| **Môn học** | CS221 - Xử lý ngôn ngữ tự nhiên |
| **Repository** | [github.com/AdamNbz/CS221](https://github.com/AdamNbz/CS221) |

---
