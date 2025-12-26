"""
Flask Web Application for INSTRUCTOR vs GTR-T5 Demo
Backend implementation based on demo.ipynb
"""

import os
import json
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

# Initialize Flask app
app = Flask(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Global variables for models
gtr_model = None
instructor_model = None
models_loaded = False

def load_models():
    """Load both models (lazy loading)"""
    global gtr_model, instructor_model, models_loaded
    
    if models_loaded:
        return True
    
    try:
        print("Loading GTR-T5 model...")
        gtr_model = SentenceTransformer('sentence-transformers/gtr-t5-large')
        print("✅ GTR-T5 loaded!")
        
        print("Loading INSTRUCTOR model...")
        instructor_model = INSTRUCTOR('hkunlp/instructor-large')
        print("✅ INSTRUCTOR loaded!")
        
        models_loaded = True
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check if models are loaded"""
    return jsonify({
        'models_loaded': models_loaded,
        'gtr_loaded': gtr_model is not None,
        'instructor_loaded': instructor_model is not None
    })

@app.route('/api/load_models', methods=['POST'])
def api_load_models():
    """Load models on demand"""
    success = load_models()
    return jsonify({'success': success, 'models_loaded': models_loaded})

# ==================== DEMO 1: Document Retrieval ====================

@app.route('/api/retrieval', methods=['POST'])
def api_retrieval():
    """Compare document retrieval between GTR-T5 and INSTRUCTOR"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 400
    
    data = request.json
    query = data.get('query', '')
    documents = data.get('documents', [])
    query_instruction = data.get('query_instruction', 'Represent the question for retrieving supporting documents:')
    doc_instruction = data.get('doc_instruction', 'Represent the document for retrieval:')
    
    if not query or not documents:
        return jsonify({'error': 'Query and documents required'}), 400
    
    # GTR-T5 encoding (no instruction)
    query_emb_gtr = gtr_model.encode([query])
    doc_embs_gtr = gtr_model.encode(documents)
    sim_gtr = cosine_similarity(query_emb_gtr, doc_embs_gtr)[0]
    
    # INSTRUCTOR encoding (with instruction)
    query_emb_inst = instructor_model.encode([[query_instruction, query]])
    doc_embs_inst = instructor_model.encode([[doc_instruction, doc] for doc in documents])
    sim_inst = cosine_similarity(query_emb_inst, doc_embs_inst)[0]
    
    # Rank documents
    ranked_gtr = sorted(enumerate(sim_gtr), key=lambda x: x[1], reverse=True)
    ranked_inst = sorted(enumerate(sim_inst), key=lambda x: x[1], reverse=True)
    
    return jsonify({
        'gtr_results': [
            {'index': int(idx), 'document': documents[idx], 'score': float(score)}
            for idx, score in ranked_gtr
        ],
        'instructor_results': [
            {'index': int(idx), 'document': documents[idx], 'score': float(score)}
            for idx, score in ranked_inst
        ],
        'similarity_matrix_gtr': sim_gtr.tolist(),
        'similarity_matrix_inst': sim_inst.tolist()
    })

# ==================== DEMO 2: Task-Specific Instructions ====================

@app.route('/api/task_embeddings', methods=['POST'])
def api_task_embeddings():
    """Show how same text produces different embeddings with different instructions"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 400
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    # Task instructions
    task_instructions = {
        'retrieval_query': 'Represent the question for retrieving relevant documents:',
        'retrieval_doc': 'Represent the document for retrieval:',
        'classification': 'Represent the sentence for classification:',
        'clustering': 'Represent the sentence for clustering:',
        'similarity': 'Represent the sentence for semantic similarity:',
    }
    
    # GTR-T5: single embedding
    emb_gtr = gtr_model.encode([text])[0]
    
    # INSTRUCTOR: different embeddings per task
    embeddings_inst = {}
    for task, instruction in task_instructions.items():
        embeddings_inst[task] = instructor_model.encode([[instruction, text]])[0]
    
    # Compute similarities between task embeddings
    tasks = list(task_instructions.keys())
    similarity_matrix = []
    
    for i, task1 in enumerate(tasks):
        row = []
        for j, task2 in enumerate(tasks):
            sim = float(cosine_similarity([embeddings_inst[task1]], [embeddings_inst[task2]])[0][0])
            row.append(sim)
        similarity_matrix.append(row)
    
    return jsonify({
        'text': text,
        'tasks': tasks,
        'task_instructions': task_instructions,
        'similarity_matrix': similarity_matrix,
        'message': 'Same text produces different embeddings with different instructions!'
    })

# ==================== DEMO 3: Clustering ====================

@app.route('/api/clustering', methods=['POST'])
def api_clustering():
    """Compare clustering performance between GTR-T5 and INSTRUCTOR"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 400
    
    data = request.json
    texts = data.get('texts', [])
    true_labels = data.get('labels', [])
    n_clusters = data.get('n_clusters', 3)
    cluster_instruction = data.get('instruction', 'Represent the text for topic clustering:')
    
    if not texts or len(texts) < 3:
        return jsonify({'error': 'At least 3 texts required'}), 400
    
    # GTR-T5 embeddings
    emb_gtr = gtr_model.encode(texts)
    
    # INSTRUCTOR embeddings
    emb_inst = instructor_model.encode([[cluster_instruction, t] for t in texts])
    
    # KMeans clustering
    kmeans_gtr = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_gtr = kmeans_gtr.fit_predict(emb_gtr)
    sil_gtr = silhouette_score(emb_gtr, pred_gtr) if len(set(pred_gtr)) > 1 else 0
    
    kmeans_inst = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_inst = kmeans_inst.fit_predict(emb_inst)
    sil_inst = silhouette_score(emb_inst, pred_inst) if len(set(pred_inst)) > 1 else 0
    
    # t-SNE for visualization
    tsne_gtr = TSNE(n_components=2, random_state=42, perplexity=min(3, len(texts)-1))
    emb_2d_gtr = tsne_gtr.fit_transform(emb_gtr)
    
    tsne_inst = TSNE(n_components=2, random_state=42, perplexity=min(3, len(texts)-1))
    emb_2d_inst = tsne_inst.fit_transform(emb_inst)
    
    return jsonify({
        'gtr': {
            'predictions': pred_gtr.tolist(),
            'silhouette': float(sil_gtr),
            'tsne_coords': emb_2d_gtr.tolist()
        },
        'instructor': {
            'predictions': pred_inst.tolist(),
            'silhouette': float(sil_inst),
            'tsne_coords': emb_2d_inst.tolist()
        },
        'texts': texts,
        'true_labels': true_labels if true_labels else None
    })

# ==================== DEMO 4: Custom Comparison ====================

@app.route('/api/compare_sentences', methods=['POST'])
def api_compare_sentences():
    """Compare similarity between two sentences using both models"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 400
    
    data = request.json
    sentence1 = data.get('sentence1', '')
    sentence2 = data.get('sentence2', '')
    instruction = data.get('instruction', 'Represent the sentence for semantic similarity:')
    
    if not sentence1 or not sentence2:
        return jsonify({'error': 'Both sentences required'}), 400
    
    # GTR-T5
    emb1_gtr = gtr_model.encode([sentence1])[0]
    emb2_gtr = gtr_model.encode([sentence2])[0]
    sim_gtr = float(cosine_similarity([emb1_gtr], [emb2_gtr])[0][0])
    
    # INSTRUCTOR
    emb1_inst = instructor_model.encode([[instruction, sentence1]])[0]
    emb2_inst = instructor_model.encode([[instruction, sentence2]])[0]
    sim_inst = float(cosine_similarity([emb1_inst], [emb2_inst])[0][0])
    
    return jsonify({
        'sentence1': sentence1,
        'sentence2': sentence2,
        'instruction': instruction,
        'gtr_similarity': sim_gtr,
        'instructor_similarity': sim_inst,
        'difference': sim_inst - sim_gtr
    })


# ==================== LOAD DEMO DATA ====================

def load_demo_data():
    """Load demo data from JSON file"""
    demo_data_path = os.path.join(os.path.dirname(__file__), 'demo_data.json')
    try:
        with open(demo_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading demo data: {e}")
        return None

# ==================== PRESET DEMOS ====================

@app.route('/api/preset/retrieval')
def preset_retrieval():
    """Get preset data for retrieval demo"""
    demo_data = load_demo_data()
    if demo_data and 'retrieval_demos' in demo_data:
        return jsonify(demo_data['retrieval_demos'])
    # Fallback
    return jsonify([{
        'name': 'Neural Networks & AI',
        'query': 'How do neural networks learn?',
        'documents': [
            'Python is a programming language known for its simple syntax.',
            'Machine learning algorithms can learn patterns from data.',
            'The Eiffel Tower is located in Paris, France.',
            'Neural networks are inspired by the human brain.',
            'Climate change is affecting global weather patterns.',
            'Deep learning is a subset of machine learning using neural networks.',
        ],
        'query_instruction': 'Represent the question for retrieving supporting documents:',
        'doc_instruction': 'Represent the document for retrieval:'
    }])

@app.route('/api/preset/clustering')
def preset_clustering():
    """Get preset data for clustering demo"""
    demo_data = load_demo_data()
    if demo_data and 'clustering_demos' in demo_data:
        return jsonify(demo_data['clustering_demos'])
    # Fallback
    return jsonify([{
        'name': 'Sports, Technology, Food (3 topics)',
        'texts': [
            'The football match ended with a dramatic goal in overtime.',
            'Basketball players need excellent jumping abilities.',
            'Tennis requires both physical and mental strength.',
            'Artificial intelligence is transforming many industries.',
            'Quantum computing promises unprecedented processing power.',
            'Cybersecurity is essential in the digital age.',
            'Italian pasta dishes are popular worldwide.',
            'Sushi is a traditional Japanese cuisine.',
            'French pastries are known for their delicate flavors.'
        ],
        'labels': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'label_names': ['Sports', 'Technology', 'Food'],
        'n_clusters': 3,
        'instruction': 'Represent the news article for topic clustering:'
    }])


@app.route('/api/preset/task_embeddings')
def preset_task_embeddings():
    """Get preset data for task embeddings demo"""
    demo_data = load_demo_data()
    if demo_data and 'task_embedding_demos' in demo_data:
        return jsonify(demo_data['task_embedding_demos'])
    # Fallback
    return jsonify([{
        'name': 'Machine Learning Statement',
        'text': 'Machine learning is transforming how we analyze data',
        'description': 'Một câu về machine learning'
    }])

@app.route('/api/preset/similarity')
def preset_similarity():
    """Get preset data for similarity demo"""
    demo_data = load_demo_data()
    if demo_data and 'similarity_demos' in demo_data:
        return jsonify(demo_data['similarity_demos'])
    # Fallback
    return jsonify([{
        'name': 'Semantic Similarity - High',
        'sentence1': 'The cat is sleeping on the couch.',
        'sentence2': 'A feline is resting on the sofa.',
        'instruction': 'Represent the sentence for semantic similarity:',
        'expected': 'high'
    }])

@app.route('/api/preset/instructions')
def preset_instructions():
    """Get reference instructions for different tasks"""
    demo_data = load_demo_data()
    if demo_data and 'instructions_reference' in demo_data:
        return jsonify(demo_data['instructions_reference'])
    return jsonify({})

# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("INSTRUCTOR vs GTR-T5 Demo Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Models will be loaded on first API call or via /api/load_models")
    print("\nOpen http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
