"""Embedding-based sentence filtering for financial metrics."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import asyncio
import logging

logger = logging.getLogger(__name__)

class EmbeddingFilter:
    """Filter sentences using semantic similarity to metric definitions."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified model.
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        
    async def filter_sentences(
        self,
        sentences: Dict[int, str],
        definitions: List[Dict],
        threshold: float = 0.4,
        timeout: int = 30
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Filter sentences by similarity to metric definitions.
        
        Args:
            sentences: Dictionary mapping indices to sentences
            definitions: List of metric definition dictionaries with structure:
                {
                    "definition": str,  # e.g., "Total revenues"
                    "values": List[str],  # e.g., ["24.93", "48.26"]
                    "variations": List[str]  # e.g., ["total revenue", "revenues"]
                }
            threshold: Minimum similarity score (default: 0.4)
            timeout: Timeout in seconds for embedding computation
            
        Returns:
            Dictionary mapping sentence indices to list of (definition, similarity_score)
        """
        try:
            # Combine all variations for each definition
            definition_texts = []
            definition_map = {}  # Map variation back to original definition
            
            for def_dict in definitions:
                definition = def_dict["definition"]
                for variation in def_dict["variations"]:
                    definition_texts.append(variation)
                    definition_map[variation] = definition
            
            # Get embeddings for definitions
            definition_embeddings = await asyncio.wait_for(
                asyncio.to_thread(self.model.encode, definition_texts),
                timeout=timeout
            )
            
            # Get embeddings for sentences
            sentence_embeddings = await asyncio.wait_for(
                asyncio.to_thread(self.model.encode, list(sentences.values())),
                timeout=timeout
            )
            
            # Calculate similarities and filter
            filtered_sentences = {}
            
            for idx, sentence_embedding in enumerate(sentence_embeddings):
                matches = []
                
                for def_idx, def_embedding in enumerate(definition_embeddings):
                    similarity = float(1 - cosine(sentence_embedding, def_embedding))
                    
                    if similarity >= threshold:
                        definition = definition_map[definition_texts[def_idx]]
                        matches.append((definition, similarity))
                
                if matches:
                    # Sort by similarity score
                    matches.sort(key=lambda x: x[1], reverse=True)
                    filtered_sentences[idx] = matches
                    
                    logger.debug(f"Sentence {idx} matched definitions:")
                    for def_name, score in matches:
                        logger.debug(f"  {def_name}: {score:.2%}")
            
            return filtered_sentences
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout ({timeout}s) exceeded during embedding computation")
            return {}
        except Exception as e:
            logger.error(f"Error in filter_sentences: {str(e)}")
            return {}
