"""
Stage 4: Anomaly Detection (Memory-First Similarity Matching)

Uses embeddings and FAISS indexing to compare motion events against historical patterns.
Filters events based on similarity to past events and support count.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pickle

try:
    import faiss
except ImportError:
    faiss = None

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

try:
    import torch
except ImportError:
    torch = None


class Event:
    """Represents a motion event with multiple motion boxes and embeddings."""
    
    def __init__(
        self,
        event_id: str,
        camera_id: str,
        timestamp: datetime,
        motion_boxes: List[Tuple[int, int, int, int]],
        embeddings: np.ndarray,
        label: Optional[str] = None,
    ):
        """
        Args:
            event_id: Unique event identifier
            camera_id: Camera that captured the event
            timestamp: When event occurred
            motion_boxes: List of (x1, y1, x2, y2) bounding boxes
            embeddings: N x d array where N = num boxes, d = embedding dim
            label: Optional label ("true_event", "false_alarm", etc.)
        """
        self.event_id = event_id
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.motion_boxes = motion_boxes
        self.embeddings = embeddings  # Shape: (N, d)
        self.label = label
        self.num_boxes = len(motion_boxes)


class MemoryIndex:
    """Per-camera FAISS index and metadata store for event embeddings."""
    
    def __init__(
        self,
        camera_id: str,
        embedding_dim: int = 512,
        storage_path: str = "data/anomaly_indices",
    ):
        self.camera_id = camera_id
        self.embedding_dim = embedding_dim
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index: IndexFlatIP for cosine similarity (normalized)
        self.index = faiss.IndexFlatIP(embedding_dim) if faiss else None
        
        # Metadata per embedding
        self.metadata: List[Dict] = []
        self.embeddings_list: List[np.ndarray] = []
    
    def add_event(self, event: Event) -> None:
        """Add all embeddings from an event to the index."""
        if self.index is None:
            raise RuntimeError("FAISS not installed")
        
        # Normalize embeddings to unit length for cosine similarity
        embeddings_norm = event.embeddings / (
            np.linalg.norm(event.embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Add to FAISS index
        self.index.add(embeddings_norm.astype(np.float32))
        
        # Store metadata for each embedding
        for i, (box, emb) in enumerate(zip(event.motion_boxes, embeddings_norm)):
            self.metadata.append({
                "event_id": event.event_id,
                "camera_id": event.camera_id,
                "timestamp": event.timestamp,
                "box": box,
                "box_idx": i,
                "label": event.label,
            })
            self.embeddings_list.append(emb)
    
    def retrieve_neighbors(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve k nearest neighbors for each query embedding.
        
        Args:
            query_embeddings: Shape (M, d) where M = num query boxes
            k: Number of neighbors to retrieve per query
        
        Returns:
            (similarities, indices) where:
            - similarities: (M, k) array of cosine similarities
            - indices: (M, k) array of FAISS index positions
        """
        if self.index is None or self.index.ntotal == 0:
            return np.zeros((query_embeddings.shape[0], k)), np.zeros((query_embeddings.shape[0], k), dtype=np.int64)
        
        # Normalize query
        query_norm = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Retrieve neighbors
        similarities, indices = self.index.search(query_norm.astype(np.float32), k)
        return similarities, indices
    
    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            return
        
        index_path = self.storage_path / f"index_{self.camera_id}.faiss"
        meta_path = self.storage_path / f"metadata_{self.camera_id}.pkl"
        
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
            }, f)
    
    def load(self) -> bool:
        """Load index and metadata from disk."""
        index_path = self.storage_path / f"index_{self.camera_id}.faiss"
        meta_path = self.storage_path / f"metadata_{self.camera_id}.pkl"
        
        if not index_path.exists() or not meta_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False


class AnomalyDetector:
    """
    Stage 4: Memory-first anomaly detection using embedding similarity.
    
    Decision rule:
    - FILTER (pass to next stage) if: similarity is high AND support is strong
    - DROP (normal event) if: similarity is low OR support is weak
    """
    
    def __init__(
        self,
        embedding_model: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 512,
        sim_strong: float = 0.92,
        sim_weak: float = 0.85,
        support_min: int = 8,
        storage_path: str = "data/anomaly_indices",
    ):
        """
        Args:
            embedding_model: HuggingFace model ID for vision embeddings
            embedding_dim: Output embedding dimension
            sim_strong: Threshold for strong similarity match
            sim_weak: Threshold below which event is novel
            support_min: Minimum number of neighbors for support
            storage_path: Directory to store FAISS indices
        """
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.sim_strong = sim_strong
        self.sim_weak = sim_weak
        self.support_min = support_min
        self.storage_path = storage_path
        
        # Load embedding model
        self.embed_processor = None
        self.embed_model = None
        if AutoModel and AutoProcessor:
            try:
                self.embed_processor = AutoProcessor.from_pretrained(embedding_model)
                self.embed_model = AutoModel.from_pretrained(embedding_model)
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
        
        # Per-camera indices
        self.indices: Dict[str, MemoryIndex] = {}
    
    def extract_embeddings(
        self,
        images: List[np.ndarray],
        motion_boxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Extract embeddings for motion boxes from image (I2).
        
        Args:
            images: List of images [I1, I2, I3] (use I2)
            motion_boxes: List of (x1, y1, x2, y2) bounding boxes
        
        Returns:
            embeddings: (N, d) array where N = num boxes
        """
        if self.embed_model is None or len(images) < 2:
            # Return dummy embeddings if model not available
            return np.random.randn(len(motion_boxes), self.embedding_dim).astype(np.float32)
        
        # Use I2 (middle frame)
        image = images[1]
        embeddings = []
        
        for (x1, y1, x2, y2) in motion_boxes:
            # Crop patch
            patch = image[y1:y2, x1:x2]
            
            if patch.size == 0:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue
            
            try:
                # Prepare input
                inputs = self.embed_processor(images=patch, return_tensors="pt")
                
                # Extract embedding
                if torch is not None:
                    with torch.no_grad():
                        outputs = self.embed_model(**inputs)
                        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        embeddings.append(emb.astype(np.float32))
                else:
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_or_create_index(self, camera_id: str) -> MemoryIndex:
        """Get or create FAISS index for camera."""
        if camera_id not in self.indices:
            self.indices[camera_id] = MemoryIndex(
                camera_id=camera_id,
                embedding_dim=self.embedding_dim,
                storage_path=self.storage_path,
            )
            # Try to load existing index
            self.indices[camera_id].load()
        return self.indices[camera_id]
    
    def detect(
        self,
        camera_id: str,
        images: List[np.ndarray],
        motion_boxes: List[Tuple[int, int, int, int]],
        top_k: int = 10,
    ) -> Dict[str, object]:
        """
        Detect if event is anomalous based on historical similarity.
        
        Args:
            camera_id: Camera identifier
            images: [I1, I2, I3]
            motion_boxes: List of motion bounding boxes
            top_k: Number of neighbors to retrieve
        
        Returns:
            Dict with detection result:
            - event_sim: Best similarity score
            - event_support: Count of strong neighbors
            - decision: "FILTER" (anomalous) or "DROP" (normal)
            - neighbors: List of matched past events
            - reasoning: Explanation string
        """
        if not motion_boxes:
            return {
                "event_sim": 0.0,
                "event_support": 0,
                "decision": "DROP",
                "neighbors": [],
                "reasoning": "No motion boxes detected",
            }
        
        # Extract embeddings
        query_embeddings = self.extract_embeddings(images, motion_boxes)
        
        # Get or create index
        index = self.get_or_create_index(camera_id)
        
        # Retrieve neighbors
        similarities, indices = index.retrieve_neighbors(query_embeddings, k=top_k)
        
        # Aggregate similarity and support
        best_sims = np.max(similarities, axis=1)  # Best match per box
        event_sim = float(np.max(best_sims)) if len(best_sims) > 0 else 0.0
        
        # Count strong matches (support)
        support_threshold = self.sim_strong - 0.05  # Slightly relaxed
        strong_matches = []
        unique_event_ids = set()
        
        for i, sims in enumerate(similarities):
            for sim, idx in zip(sims, indices[i]):
                if sim >= support_threshold and idx >= 0 and idx < len(index.metadata):
                    metadata = index.metadata[int(idx)]
                    strong_matches.append({
                        "event_id": metadata["event_id"],
                        "similarity": float(sim),
                        "timestamp": metadata["timestamp"],
                        "label": metadata["label"],
                    })
                    unique_event_ids.add(metadata["event_id"])
        
        event_support = len(unique_event_ids)  # Count unique past events
        
        # Decision rule
        if event_sim >= self.sim_strong and event_support >= self.support_min:
            decision = "FILTER"  # Anomalous → pass to YOLO
            reasoning = f"Strong similarity ({event_sim:.3f}) with {event_support} past events"
        elif event_sim < self.sim_weak:
            decision = "DROP"  # Novel → normal
            reasoning = f"Low similarity ({event_sim:.3f}); novel event"
        elif event_support < 3:
            decision = "DROP"  # No historical support
            reasoning = f"Weak support ({event_support} matches); likely false alarm"
        else:
            decision = "PASS"  # Uncertain → may go to Stage 5
            reasoning = f"Moderate match ({event_sim:.3f}) with {event_support} neighbors"
        
        return {
            "event_sim": event_sim,
            "event_support": event_support,
            "decision": decision,
            "neighbors": strong_matches[:5],  # Top 5 matches
            "reasoning": reasoning,
        }
    
    def add_event_to_memory(
        self,
        camera_id: str,
        event_id: str,
        images: List[np.ndarray],
        motion_boxes: List[Tuple[int, int, int, int]],
        label: Optional[str] = None,
    ) -> None:
        """Add a new event to memory index."""
        # Extract embeddings
        embeddings = self.extract_embeddings(images, motion_boxes)
        
        # Create event object
        event = Event(
            event_id=event_id,
            camera_id=camera_id,
            timestamp=datetime.now(),
            motion_boxes=motion_boxes,
            embeddings=embeddings,
            label=label,
        )
        
        # Add to index
        index = self.get_or_create_index(camera_id)
        index.add_event(event)
        index.save()
    
    def get_statistics(self, camera_id: str) -> Dict:
        """Get statistics about index for a camera."""
        index = self.get_or_create_index(camera_id)
        unique_events = len(set(m["event_id"] for m in index.metadata))
        
        return {
            "camera_id": camera_id,
            "total_embeddings": len(index.metadata),
            "unique_events": unique_events,
            "index_size": index.index.ntotal if index.index else 0,
            "embedding_dim": self.embedding_dim,
        }
