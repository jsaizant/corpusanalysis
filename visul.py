from tools import *
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import gaussian_kde

COLORS = ["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "white", "yellow"]

def entropy_barplot(corpus: list, mark_outliers: bool = True):
    # Get lists
    id_list = [str(doc["id"]) for doc in corpus]
    entropy_list = [doc["entropy"] for doc in corpus]

    # Create a list of tuples with ID and entropy
    id_entropy_list = list(zip(id_list, entropy_list))

    # Sort the list of tuples by entropy
    sorted_id_entropy_list = sorted(id_entropy_list, key=lambda x: x[1])

    # Get the sorted lists of ID and entropy
    sorted_id_list = [x[0] for x in sorted_id_entropy_list]
    sorted_entropy_list = [x[1] for x in sorted_id_entropy_list]

    # Check outlier flag
    if mark_outliers:
        outliers = find_outliers({doc['id']:doc["entropy"] for doc in corpus})
        print(f"[INFO] Found {len(outliers)} entropy outliers.")
    else:
        outliers = []

    # Create a list of colors based on whether the entropy value is an outlier or not
    colors = ["red" if str(id) in outliers else "lightblue" for id in sorted_id_list]

    # Create a figure and an axis
    fig, ax = plt.subplots()

    # Plot a bar chart with the sample names and entropy values and colors
    ax.bar(sorted_id_list, sorted_entropy_list, color=colors)

    # Add some labels and a title
    ax.set_xlabel('Document')
    ax.set_ylabel('Entropy')
    ax.set_title('Comparison of entropy values across different documents')

    # Show the plot
    plt.show()

def entropy_histogram(corpus: list, lang: str = "mx"):
    # Get data list
    entropy_list = [doc["entropy"] for doc in corpus]

    fig, ax = plt.subplots()

    # Plot a bar chart with the sample names and entropy values and colors
    ax.hist(entropy_list, alpha=0.5, color=random.sample(COLORS, 1)[0], label=lang, bins=20)
        
    # Add some labels and a title
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Distribution of entropy values across a corpus sample')
    ax.legend(loc="upper right")

    plt.show()

def entropy_density_curves_corpora(corpus_list: list, pos: bool = False, ngram: int = None):
    """
    Input: List of corpora with document dictionaries 
    and entropy values computed for each document
    """
    # Create a figure and an axis object
    fig, ax = plt.subplots()
    # Loop through the lists of entropy coefficients
    for corpus in corpus_list:
        corpus = get_entropy_info(corpus, pos=pos, ngram=ngram)
        entropy_data = [doc["entropy"] for doc in corpus]
        # Estimate the density function using gaussian_kde
        density = gaussian_kde(entropy_data)
        # Generate a range of x values for plotting
        xs = np.linspace(min(entropy_data), max(entropy_data), 200)
        # Plot the density curve on the axis
        ax.plot(xs, density(xs), label=corpus[0]["source"])
    # Add a legend and labels
    ax.legend()
    ax.set_xlabel("Entropy coefficient")
    ax.set_ylabel("Density")
    # Show the plot
    plt.show()

def cosine_similarity_corpora(corpus_list: list, pos: bool = False, ngram: int = None):
    """
    Input: List of corpora with document dictionaries 
    and entropy values computed for each document
    """
    labels = []
    documents = []
    for corpus in corpus_list:
        labels.extend([document["source"]+str(i) if str(i).endswith('50') else '' for i, document in enumerate(corpus)])
        documents.extend(corpus)
    cosine_matrix = cosine_similarity(documents, pos=pos, ngram=ngram)

    # Plot the cosine similarity matrix as a heatmap with labels
    plt.figure(figsize=(8,8))
    plt.imshow(cosine_matrix, cmap="RdPu")
    plt.colorbar()
    plt.xticks(np.arange(len(documents)), labels, rotation=90)
    plt.yticks(np.arange(len(documents)), labels)
    plt.title("Cosine similarity matrix")
    plt.show()

def document_clustering_corpora(corpus_list: list, return_cluster: int = None, pos: bool = False, ngram: int = None):
    documents = []
    corpus_markers = {}  # Store corpus markers
    markers = ["8", "s", "p", "P", "*", "X", "d"]
    
    for i, corpus in enumerate(corpus_list):
        documents.extend(corpus)
        corpus_markers[i] = markers[i]  # Assign a unique marker to each corpus

    vectors_reduced, clusters, kmeans = cluster_kmeans(documents, pos=pos, ngram=ngram)

    fig, ax = plt.subplots()
    # Plot the data points with different markers for each corpus
    for i, corpus in enumerate(corpus_list):
        corpus_indices = np.arange(len(documents))[(i * len(corpus)) : ((i + 1) * len(corpus))]
        ax.scatter(
            x=vectors_reduced[corpus_indices, 0],
            y=vectors_reduced[corpus_indices, 1],
            c=clusters[corpus_indices],
            s=10, # size
            cmap="viridis", # colour map
            marker=corpus_markers[i], # symbol
            label=corpus[0]["source"]
        )

        if return_cluster: # TODO: export subcorpus cluster
            corpus_clusters = clusters[corpus_indices]
            for i in set(corpus_clusters):
                doc_cluster_lst = []
                for j in corpus_clusters:
                    if i == j:
                        doc_cluster_lst.append(corpus[j])
                with open(f"doc_cluster{i}.jsonl", "w") as fout:
                    fout.write(json.dumps(subcorpus, indent=1))

                
    # Plot the centroids
    ax.scatter(
        x=kmeans.cluster_centers_[:, 0], 
        y=kmeans.cluster_centers_[:, 1], 
        c="black", # colour
        s=400,  # size
        alpha=0.5 # transparency
        )

    # Create legend and grid
    ax.legend()

    # Show the plot
    plt.show()


def old_document_clustering_corpora(corpus_list: list, pos: bool = False, ngram: int = None): # TODO: Mark each document with the source corpus
    labels = []
    documents = []
    for corpus in corpus_list:
        documents.extend(corpus)

    vectors_reduced, clusters, kmeans = cluster_kmeans(documents, pos=pos, ngram=ngram)

    # Plot the reduced data points
    plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=clusters, s=2, cmap="viridis")

    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", s=400, alpha=0.5)

    # Show the plot
    plt.show()


def plot_confusion_matrix(confusion_dict, title):
    # Extract values from the dictionary
    tp = confusion_dict.get('TP', 0)
    fp = confusion_dict.get('FP', 0)
    tn = confusion_dict.get('TN', 0)
    fn = confusion_dict.get('FN', 0)

    # Create the confusion matrix as a numpy array
    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    

    # Set up the plot
    labels = ['Is lyrical', 'Is not lyrical', 'Is lyrical', 'Is not lyrical']
    colors = ['lightblue', 'lightcoral', 'lightcoral', 'lightblue']

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color='black', fontsize=14)

    plt.xticks([0, 1], labels[:2], rotation=45)
    plt.yticks([0, 1], labels[2:], rotation=45)
    plt.xlabel('Prediction')
    plt.ylabel('Standard')
    plt.title(title)
    plt.tight_layout()
    plt.show()