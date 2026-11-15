"""
BioMedCLIP-based image retrieval for RAG-augmented VQA.

Usage:
    rag = BioMedCLIPRAG()
    rag.load_model()
    rag.build_index(train_annotations, vis_root, cache_dir='/content/rag_cache')
    results = rag.retrieve('/path/to/test_image.jpg', top_k=3)
    context = BioMedCLIPRAG.format_context(results)
"""

import os
import torch
import numpy as np
from PIL import Image


class BioMedCLIPRAG:
    MODEL_ID = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
        self.index = None
        self.train_records = []  # list of {"question", "answer", "img_id"}

    def load_model(self):
        from open_clip import create_model_from_pretrained
        self.model, self.preprocess = create_model_from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device).eval()
        print(f'BioMedCLIP loaded on {self.device}')

    @torch.no_grad()
    def _encode_images(self, paths, batch_size=32):
        from tqdm import tqdm
        all_feats = []
        for i in tqdm(range(0, len(paths), batch_size), desc='Encoding images', unit='batch'):
            batch = paths[i : i + batch_size]
            imgs = torch.stack([
                self.preprocess(Image.open(p).convert('RGB'))
                for p in batch
            ]).to(self.device)
            with torch.amp.autocast('cuda'):
                feats = self.model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.float().cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)

    def build_index(self, train_annotations, vis_root, cache_dir=None):
        """
        Build FAISS inner-product index (= cosine similarity after L2-norm).
        train_annotations: list of dicts with 'img_id', 'question', 'answer'
        cache_dir: optional path to save/load pre-computed features
        """
        import faiss

        self.train_records = [
            {'question': a['question'], 'answer': a['answer'], 'img_id': a['img_id']}
            for a in train_annotations
        ]

        feat_cache = os.path.join(cache_dir, 'train_feats.npy') if cache_dir else None
        if feat_cache and os.path.exists(feat_cache):
            print(f'캐시 로드: {feat_cache}')
            features = np.load(feat_cache)
        else:
            paths = [os.path.join(vis_root, a['img_id']) for a in train_annotations]
            features = self._encode_images(paths)
            if feat_cache:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(feat_cache, features)
                print(f'피처 저장: {feat_cache}')

        dim = features.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(features)
        print(f'FAISS 인덱스 완성: {self.index.ntotal}개, dim={dim}')

    @torch.no_grad()
    def retrieve(self, query_image_path, top_k=3):
        """Return list of top_k most similar training records."""
        img = Image.open(query_image_path).convert('RGB')
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.amp.autocast('cuda'):
            feat = self.model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        query = feat.float().cpu().numpy().astype(np.float32)
        k = min(top_k, self.index.ntotal)
        _, indices = self.index.search(query, k)
        return [self.train_records[i] for i in indices[0] if 0 <= i < len(self.train_records)]

    @staticmethod
    def format_context(retrieved):
        """Format retrieved records as a newline-separated context string."""
        return '\n'.join(f"Q: {r['question']} A: {r['answer']}" for r in retrieved)
