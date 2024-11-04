#(First Script - PDF Processing

import spacy
from pathlib import Path
import os
import pdfplumber
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
load_dotenv()

def inspect_directory(directory_path):
    print("Analyzing directory contents...")
    pdf_files = list(Path(directory_path).glob('**/*.pdf'))
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf_path in pdf_files:
        print(f"- {pdf_path.name}")
    
    return pdf_files

def process_pdfs(directory_path):
    pdf_files = list(Path(directory_path).glob('**/*.pdf'))
    processed_docs = []
    
    print(f"\nProcessing {len(pdf_files)} PDF files...")
    
    for pdf_path in tqdm(pdf_files):
        try:
            doc_content = []
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        doc_content.append({
                            'page_num': page_num,
                            'content': text.strip()
                        })
            doc_entry = {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'total_pages': len(doc_content),
                'pages': doc_content
            }
            
            processed_docs.append(doc_entry)
            
        except Exception as e:
            print(f"\nError processing {pdf_path.name}: {str(e)}")
    
    return processed_docs

def analyze_docs(processed_docs):
    print("\nDocument Analysis:")
    print(f"Total documents processed: {len(processed_docs)}")

    total_pages = sum(doc['total_pages'] for doc in processed_docs)
    avg_pages = total_pages / len(processed_docs) if processed_docs else 0
    
    print(f"Total pages: {total_pages}")
    print(f"Average pages per document: {avg_pages:.2f}")
    
    print("\nDocument previews:")
    for doc in processed_docs[:3]:  
        print(f"\nFilename: {doc['filename']}")
        print(f"Pages: {doc['total_pages']}")
        if doc['pages']:
            preview = doc['pages'][0]['content'][:200] 
            print(f"Preview: {preview}...")

def main():
    directory = os.getenv("FILE_PATH")
    output_file = "processed_user_guides.json"

    pdf_files = inspect_directory(directory)
    
    if not pdf_files:
        print("No PDF files found in directory!")
        return

    processed_docs = process_pdfs(directory)

    analyze_docs(processed_docs)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed documents saved to {output_file}")

if __name__ == "__main__":
    main()


def load_and_prepare_docs():
    print("Loading processed documents...")
    with open("processed_user_guides.json", 'r', encoding='utf-8') as f:
        docs = json.load(f)
    prepared_docs = []
    for doc in docs:
        full_text = ' '.join(page['content'] for page in doc['pages'])
        prepared_docs.append({
            'filename': doc['filename'],
            'text': full_text,
            'total_pages': doc['total_pages']
        })
    
    return prepared_docs

def extract_key_terms(docs, n_terms=20):

    print("\nExtracting key terms...")

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)  
    )
    

    tfidf_matrix = vectorizer.fit_transform([doc['text'] for doc in docs])

    feature_names = vectorizer.get_feature_names_out()

    avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    top_indices = avg_scores.argsort()[-n_terms:][::-1]
    top_terms = [(feature_names[i], avg_scores[i]) for i in top_indices]
    
    return top_terms, vectorizer, tfidf_matrix

def identify_topics(tfidf_matrix, n_clusters=5):

    print("\nIdentifying main topics...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    return clusters

def analyze_document_similarity(docs, tfidf_matrix):

    print("\nAnalyzing document similarities...")

    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    similar_pairs = []
    n_docs = len(docs)
    
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            similarity = similarity_matrix[i][j]
            if similarity > 0.2:  
                similar_pairs.append({
                    'doc1': docs[i]['filename'],
                    'doc2': docs[j]['filename'],
                    'similarity': similarity
                })

    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_pairs[:10]  

def main():

    docs = load_and_prepare_docs()
    
    top_terms, vectorizer, tfidf_matrix = extract_key_terms(docs)

    clusters = identify_topics(tfidf_matrix)

    similar_docs = analyze_document_similarity(docs, tfidf_matrix)

    print("\nAnalysis Results:")
    
    print("\nTop 20 Key Terms (with TF-IDF scores):")
    for term, score in top_terms:
        print(f"- {term}: {score:.4f}")
    
    print("\nDocument Clusters:")
    for i, doc in enumerate(docs):
        print(f"- {doc['filename']}: Cluster {clusters[i]}")
    
    print("\nMost Similar Document Pairs:")
    for pair in similar_docs:
        print(f"- {pair['doc1']} & {pair['doc2']}: {pair['similarity']:.4f} similarity")
 
    analysis_results = {
        'key_terms': [(term, float(score)) for term, score in top_terms],
        'document_clusters': [
            {'filename': doc['filename'], 'cluster': int(cluster)}
            for doc, cluster in zip(docs, clusters)
        ],
        'similar_documents': similar_docs
    }
    
    with open('document_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("\nAnalysis results saved to 'document_analysis.json'")

if __name__ == "__main__":
    main()