import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import LdaMulticore
from gensim import corpora
import re
import string
from nltk.corpus import stopwords
import nltk
import json
nltk.download('stopwords')

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if len(word) > 4]
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return tokens

def analyze_documents(json_file="processed_user_guides.json"):
    print("Loading documents...")
    with open(json_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    texts = []
    titles = []
    for doc in docs:
        text = ' '.join(page['content'] for page in doc['pages'])
        texts.append(text)
        titles.append(doc['filename'])
    
    print("Processing documents...")
    processed_texts = []
    for text in texts:
        processed = clean_doc(text)
        processed_texts.append(' '.join(processed))

    print("Creating TFIDF matrix...")
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(processed_texts)
    
    print("Performing clustering...")
    k = 8 
    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(tfidf_matrix)

    terms = tfidf.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_terms = {}
    cluster_docs = {}
    
    print("\nCluster Analysis:")
    for i in range(k):
        print(f"\nCluster {i}:")
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_terms[i] = top_terms
        print("Top terms:", ', '.join(top_terms))

        cluster_docs[i] = [titles[j] for j, c in enumerate(clusters) if c == i]
        print("Documents:", len(cluster_docs[i]))

    print("\nPerforming topic modeling...")
    processed_docs = [clean_doc(text) for text in texts]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = LdaMulticore(
        corpus,
        num_topics=5,
        id2word=dictionary,
        passes=10,
        workers=2
    )
    
    print("\nTopic Analysis:")
    for idx, topic in lda_model.print_topics(-1):
        print(f'\nTopic: {idx}')
        print(topic)

    results = {
        'cluster_analysis': {
            'terms': cluster_terms,
            'documents': cluster_docs
        },
        'topics': [
            {
                'topic_id': idx,
                'terms': topic
            } 
            for idx, topic in lda_model.print_topics(-1)
        ]
    }
    
    with open('document_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = analyze_documents()

def plot_clusters(tfidf_matrix, clusters, titles, output_prefix="cluster_viz"):

    cluster_names = {
        0: "Construction Specs",
        1: "Project Management",
        2: "Safety & Resources",
        3: "Financial Management",
        4: "Document Processing",
        5: "Security Access",
        6: "Invoice Processing",
        7: "Communication Tools"
    }
    

    n_clusters = len(set(clusters))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    print("Creating MDS visualization...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    dist = 1 - cosine_similarity(tfidf_matrix)
    pos = mds.fit_transform(dist)
    
    plt.figure(figsize=(15, 10))
    for i in range(n_clusters):
        cluster_points = pos[clusters == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[i]],
            label=cluster_names[i],
            alpha=0.6
        )
    
    plt.title("Document Clusters by Functional Area (MDS)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mds.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    pos_tsne = tsne.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(15, 10))
    for i in range(n_clusters):
        cluster_points = pos_tsne[clusters == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[i]],
            label=cluster_names[i],
            alpha=0.6
        )
    
    plt.title("Document Clusters by Functional Area (t-SNE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_documents(json_file="processed_user_guides.json"):

    print("Loading documents...")
    with open(json_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    texts = []
    titles = []
    for doc in docs:
        text = ' '.join(page['content'] for page in doc['pages'])
        texts.append(text)
        titles.append(doc['filename'])

    print("Processing documents...")
    processed_texts = []
    for text in texts:
        processed = clean_doc(text)
        processed_texts.append(' '.join(processed))

    print("Creating TFIDF matrix...")
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(processed_texts)

    print("Performing clustering...")
    k = 8 
    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(tfidf_matrix)

    plot_clusters(tfidf_matrix, clusters, titles)
    

    results['visualizations'] = {
        'mds_plot': 'cluster_viz_mds.png',
        'tsne_plot': 'cluster_viz_tsne.png',
        'heatmap': 'cluster_viz_heatmap.png'
    }
    
    return results

def main():
    print("Starting document analysis...")
    results = analyze_documents()
    

    with open('document_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAnalysis complete. Results saved to:")
    print("- document_analysis_results.json")
    print("- cluster_viz_mds.png")
    print("- cluster_viz_tsne.png")
    print("- cluster_viz_heatmap.png")

if __name__ == "__main__":
    main()