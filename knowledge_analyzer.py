#(Third Script - Analysis & Ontology)
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import spacy
from typing import Dict, List, Any
import networkx as nx
from pathlib import Path

class KnowledgeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        self.domain_concepts = {
            "software_systems": [
                "cmic", "phoenix", "mimecast", "sitedocs", "bluebeam", 
                "docusign", "globalprotect", "okta"
            ],
            "business_processes": [
                "budget", "invoice", "forecast", "closeout", "submittal",
                "rfi", "change order", "purchase order"
            ],
            "project_management": [
                "schedule", "milestone", "task", "timeline", "project",
                "planning", "resource", "assignment"
            ],
            "safety_compliance": [
                "safety", "compliance", "regulation", "certification",
                "training", "inspection", "hazard", "protection"
            ],
            "document_types": [
                "guide", "manual", "procedure", "policy", "template",
                "form", "report", "checklist"
            ]
        }
        
        self.load_documents()

    def load_documents(self):
        try:
            with open("processed_user_guides.json", 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
            print(f"Loaded {len(self.docs)} documents for analysis")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

    def build_ontology(self) -> Dict:
        print("Building ontology...")
        ontology = defaultdict(lambda: defaultdict(list))
        
        for doc in self.docs:
            text = ' '.join(page['content'] for page in doc['pages'])
            doc_nlp = self.nlp(text.lower())
            
            for category, terms in self.domain_concepts.items():
                found_terms = []
                for term in terms:
                    if term in text.lower():
                        found_terms.append(term)
                
                if found_terms:
                    ontology[category][doc['filename']] = found_terms
        
        return dict(ontology)

    def analyze_document_relationships(self) -> Dict:
        print("Analyzing document relationships...")
        
        texts = [' '.join(page['content'] for page in doc['pages']) 
                for doc in self.docs]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        G = nx.Graph()
        
        for i in range(len(self.docs)):
            for j in range(i+1, len(self.docs)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.2: 
                    G.add_edge(self.docs[i]['filename'], 
                             self.docs[j]['filename'], 
                             weight=similarity)
        
        communities = list(nx.community.greedy_modularity_communities(G))
        
        return {
            "communities": [list(c) for c in communities],
            "central_documents": list(nx.degree_centrality(G).items())
        }

    def extract_key_concepts(self) -> Dict:
        print("Extracting key concepts...")
        
        concepts = defaultdict(int)
        concept_relationships = defaultdict(lambda: defaultdict(int))
        
        for doc in self.docs:
            text = ' '.join(page['content'] for page in doc['pages'])
            doc_nlp = self.nlp(text)
            
            for chunk in doc_nlp.noun_chunks:
                if len(chunk.text.split()) <= 3:  
                    concepts[chunk.text.lower()] += 1
            
            for sent in doc_nlp.sents:
                sent_concepts = [chunk.text.lower() 
                               for chunk in sent.noun_chunks 
                               if len(chunk.text.split()) <= 3]
                
                for i in range(len(sent_concepts)):
                    for j in range(i+1, len(sent_concepts)):
                        concept_relationships[sent_concepts[i]][sent_concepts[j]] += 1
        
        return {
            "concepts": dict(concepts),
            "relationships": dict(concept_relationships)
        }

    def analyze_document_coverage(self) -> Dict:
        print("Analyzing document coverage...")
        
        coverage = defaultdict(list)
        
        for doc in self.docs:
            text = ' '.join(page['content'] for page in doc['pages'])
            
            for category, terms in self.domain_concepts.items():
                coverage_score = sum(1 for term in terms if term in text.lower())
                if coverage_score > 0:
                    coverage[category].append({
                        'filename': doc['filename'],
                        'coverage_score': coverage_score / len(terms)
                    })
        
        return dict(coverage)

    def generate_analysis_report(self) -> Dict:
        print("Generating analysis report...")
        
        ontology = self.build_ontology()
        relationships = self.analyze_document_relationships()
        concepts = self.extract_key_concepts()
        coverage = self.analyze_document_coverage()
        
        analysis_report = {
            "ontology": ontology,
            "document_relationships": relationships,
            "key_concepts": concepts,
            "coverage_analysis": coverage,
            "statistics": {
                "total_documents": len(self.docs),
                "document_communities": len(relationships["communities"]),
                "key_concepts_found": len(concepts["concepts"]),
                "domain_coverage": {
                    category: len(docs) 
                    for category, docs in coverage.items()
                }
            }
        }
        
        with open('knowledge_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2)
        
        return analysis_report

def main():
    analyzer = KnowledgeAnalyzer()
    report = analyzer.generate_analysis_report()
    print("\nAnalysis Summary:")
    print(f"Total Documents: {report['statistics']['total_documents']}")
    print(f"Document Communities: {report['statistics']['document_communities']}")
    print("\nDomain Coverage:")
    for domain, count in report['statistics']['domain_coverage'].items():
        print(f"- {domain}: {count} documents")

if __name__ == "__main__":
    main()