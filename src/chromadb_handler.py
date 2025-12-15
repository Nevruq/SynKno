import chromadb
import numpy as np
import torch
from transformers import pipeline
import bm25s
from transformers import BertTokenizer
import re


class ChromaDBHandler:
    def __init__(self, collection):
        
        self.tokenized_docs = None
        self.collection = collection
        idx_and_ids = self.get_idx2id()
        self.id2idx = idx_and_ids[0]
        self.idx2id = idx_and_ids[1]

        docs = self.get_infos_collection()[0]
        self.retriever = bm25s.BM25(corpus=docs)
        self.retriever.index(bm25s.tokenize(docs))
    

    def return_k_query_results(self, query):
        return self.retriever.retrieve(bm25s.tokenize(query), k=3)

    def bm25_tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()


    def get_infos_collection(self):
        all_docs = self.collection.get()   # careful: if you have millions, you may need paging
        doc_texts = all_docs["documents"]
        doc_ids = all_docs["ids"]
        return doc_texts, doc_ids


    def get_idx2id(self):
        """
        Returns: id2idx, idx2id
        """
            # Get all documents + ids from Chroma
            #   # careful: if you have millions, you may need paging

        doc_texts, doc_ids = self.get_infos_collection() 
        

        # Simple tokenizer – you can plug in something better if you like

        self.tokenized_docs = [self.bm25_tokenize(doc) for doc in doc_texts]

        # mappings between ids and index positions
        id2idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        idx2id = {i: doc_id for i, doc_id in enumerate(doc_ids)}

        return id2idx, idx2id

    def hybrid_retrieve(self, query: str , k: int = 5, alpha: float = 0.6):
        """
        query  : user question
        k      : number of final docs to return
        alpha  : weight for vector similarity (0..1),
                (1 - alpha) is weight for BM25
        """

        doc_texts, doc_ids = self.get_infos_collection() 
        # --- BM25 part ---
        tokenized_query = self.bm25_tokenize(query)
        bm25 = BM25Okapi(self.tokenized_docs)

        bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=float)  # len = num_docs

        # normalize BM25 scores to 0–1
        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        # --- Vector DB part (Chroma) ---
        vec_results = self.collection.query(
            query_texts=[query],
            n_results=k * 3,  # ask for more, we’ll re-rank
        )

        vec_ids = vec_results["ids"][0]
        vec_distances = np.array(vec_results["distances"][0], dtype=float)

        # convert distances to similarity (simple transform)
        vec_scores = 1.0 / (1.0 + vec_distances)

        # normalize vector scores to 0–1
        if vec_scores.max() > 0:
            vec_scores /= vec_scores.max()

        # --- combine scores ---
        combined = {}

        # start with BM25 scores for ALL documents
        for idx, b_score in enumerate(bm25_scores):
            doc_id = self.idx2id[idx]
            combined[doc_id] = (1 - alpha) * b_score

        # add / update with vector scores for hits
        for doc_id, v_score in zip(vec_ids, vec_scores):
            combined[doc_id] = combined.get(doc_id, 0.0) + alpha * v_score

        # sort by combined score
        ranked_ids = sorted(combined.keys(), key=lambda d: combined[d], reverse=True)[:k]

        ranked_docs = [doc_texts[self.id2idx[d]] for d in ranked_ids]
        ranked_scores = [combined[d] for d in ranked_ids]

        return {
            "ids": [ranked_ids],
            "documents": [ranked_docs],
            "scores": [ranked_scores],
        }