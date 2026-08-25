import pytest
import os
from retriever import Retriever, DocumentChunk

def test_retriever_indexing():
    # Load default retriever over knowledge-base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_dir = os.path.join(base_dir, 'knowledge-base')
    retriever = Retriever(kb_dir)
    
    assert len(retriever.chunks) > 0
    # Verify that front matter metadata is captured
    first_chunk = retriever.chunks[0]
    assert first_chunk.filename is not None
    assert first_chunk.status in ['active', 'superseded', 'draft']

def test_retriever_ranking_precedence():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_dir = os.path.join(base_dir, 'knowledge-base')
    retriever = Retriever(kb_dir)
    
    # Query matching standard return policy terms
    # Verify active source is boosted over superseded ones
    results = retriever.retrieve("What is the standard return window?")
    assert len(results) > 0
    
    # The top result should be the current returns policy document
    top_chunk, score = results[0]
    assert top_chunk.filename == "01-returns-policy-current.md"
    assert top_chunk.status == "active"
