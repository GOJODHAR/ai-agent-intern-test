import os
import re
import math
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class DocumentChunk:
    chunk_id: str         # document_id + heading
    document_id: str
    filename: str
    doc_title: str
    heading: str
    content: str
    status: str
    policy_authority: str
    audience: str
    effective_date: str
    last_reviewed: str
    supersedes: str = None
    superseded_by: str = None

class TFIDFIndex:
    def __init__(self):
        self.doc_frequencies: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {} # chunk_id -> term -> count
        self.num_docs = 0
        self.avg_doc_len = 0.0

    def tokenize(self, text: str) -> List[str]:
        # Lowercase and keep alphanumeric and spaces
        text = text.lower()
        # Replace non-alphanumeric characters with spaces
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 1] # ignore single character tokens unless necessary

    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenize(text)
        if not tokens:
            return
        
        self.num_docs += 1
        self.doc_lengths[doc_id] = len(tokens)
        
        # Count term frequencies for this document
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        self.term_frequencies[doc_id] = tf
        
        # Update document frequencies
        for token in tf.keys():
            self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1

    def calculate_idf(self, term: str) -> float:
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        # Standard IDF formula
        return math.log(1.0 + (self.num_docs / df))

    def get_score(self, query_tokens: List[str], doc_id: str) -> float:
        if doc_id not in self.term_frequencies:
            return 0.0
        
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        tf_dict = self.term_frequencies[doc_id]
        
        for token in query_tokens:
            if token in tf_dict:
                # Basic TF-IDF calculation
                # TF normalized by doc length
                tf = tf_dict[token] / doc_len
                idf = self.calculate_idf(token)
                score += tf * idf
                
                # Boost for exact word matches in query
                if len(token) > 3:
                    score += 0.05
        return score

class Retriever:
    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        self.chunks: List[DocumentChunk] = []
        self.index = TFIDFIndex()
        self.load_and_index()

    def parse_markdown_file(self, filepath: str) -> List[DocumentChunk]:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split metadata and body
        parts = content.split('---', 2)
        if len(parts) < 3:
            # No front matter found
            metadata = {}
            body = content
        else:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception:
                metadata = {}
            body = parts[2]

        doc_id = metadata.get('document_id', filename)
        doc_title = metadata.get('title', filename)
        status = metadata.get('status', 'active')
        policy_authority = metadata.get('policy_authority', 'official')
        audience = metadata.get('audience', 'customer')
        effective_date = str(metadata.get('effective_date', ''))
        last_reviewed = str(metadata.get('last_reviewed', ''))
        supersedes = metadata.get('supersedes')
        superseded_by = metadata.get('superseded_by')

        chunks: List[DocumentChunk] = []
        
        # Split body by markdown headers (## )
        # Regex matches lines starting with '## ' and captures the heading text
        sections = re.split(r'\n##\s+', '\n' + body.strip())
        
        # The first section contains content before any '## ' (like Overview or Title)
        intro_content = sections[0].strip()
        if intro_content:
            # Clean up title if it contains '# '
            intro_lines = intro_content.split('\n')
            heading = "Overview"
            for line in intro_lines:
                if line.startswith('# '):
                    heading = line.replace('# ', '').strip()
                    break
            
            chunk_id = f"{doc_id}_Overview"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=filename,
                doc_title=doc_title,
                heading=heading,
                content=intro_content,
                status=status,
                policy_authority=policy_authority,
                audience=audience,
                effective_date=effective_date,
                last_reviewed=last_reviewed,
                supersedes=supersedes,
                superseded_by=superseded_by
            ))

        for section in sections[1:]:
            section = section.strip()
            if not section:
                continue
            
            # Split heading from content
            lines = section.split('\n', 1)
            heading = lines[0].strip()
            sect_content = lines[1].strip() if len(lines) > 1 else ""
            
            chunk_id = f"{doc_id}_{heading.replace(' ', '_')}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=filename,
                doc_title=doc_title,
                heading=heading,
                content=f"## {heading}\n\n{sect_content}",
                status=status,
                policy_authority=policy_authority,
                audience=audience,
                effective_date=effective_date,
                last_reviewed=last_reviewed,
                supersedes=supersedes,
                superseded_by=superseded_by
            ))

        return chunks

    def load_and_index(self):
        if not os.path.exists(self.kb_dir):
            return
        
        for file in os.listdir(self.kb_dir):
            if file.endswith('.md'):
                filepath = os.path.join(self.kb_dir, file)
                file_chunks = self.parse_markdown_file(filepath)
                for chunk in file_chunks:
                    self.chunks.append(chunk)
                    # Index using the text content (including metadata keywords for richer context)
                    index_text = f"{chunk.doc_title} {chunk.heading} {chunk.content}"
                    self.index.add_document(chunk.chunk_id, index_text)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        query_tokens = self.index.tokenize(query)
        if not query_tokens:
            # If query tokens are empty, return top_k chunks from active/official sources as placeholder
            active_chunks = [c for c in self.chunks if c.status == 'active' and c.policy_authority == 'official']
            return [(c, 0.0) for c in active_chunks[:top_k]]

        scored_chunks = []
        for chunk in self.chunks:
            # 1. TF-IDF Score
            base_score = self.index.get_score(query_tokens, chunk.chunk_id)
            
            # If the chunk has some keyword overlap, apply metadata ranking
            if base_score > 0.0:
                reranked_score = base_score
                
                # Bonus for active, official documents
                if chunk.status == 'active' and chunk.policy_authority == 'official':
                    reranked_score += 2.0
                
                # Penalty for superseded documents
                if chunk.status == 'superseded':
                    reranked_score -= 5.0
                
                # Penalty for draft/non-policy documents
                if chunk.status == 'draft' or chunk.policy_authority == 'none':
                    reranked_score -= 5.0
                    
                # Audience bonus: slight preference for customer over internal content
                if chunk.audience == 'customer':
                    reranked_score += 0.2
                
                # Escalation keyword boosting for escalation document
                is_escalation_query = any(q in query.lower() for q in ['escalat', 'human', 'specialist', 'agent limit', 'handoff'])
                if chunk.filename == '13-support-escalation.md':
                    if is_escalation_query:
                        reranked_score += 1.5
                    else:
                        # Slight penalty to internal escalation rules for normal queries
                        reranked_score -= 0.5

                # Boost exact match of key product names or unique terms
                # e.g., "breeze tumbler", "trailplus", "canada"
                lower_content = chunk.content.lower() + " " + chunk.heading.lower() + " " + chunk.doc_title.lower()
                for keyword in ['breeze tumbler', 'trailplus', 'canada', 'germany', 'warranty', 'cancellation']:
                    if keyword in query.lower() and keyword in lower_content:
                        reranked_score += 0.5
                
                scored_chunks.append((chunk, reranked_score))
        
        # Sort by reranked score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
