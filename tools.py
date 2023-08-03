from read import *
import nltk
import re
import random
import numpy as np
from math import log2
from langdetect import detect_langs, detect
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

def iso_to_english(code: str):
    # Dictionary from english to ISO 639-1
    iso639_to_languages = {
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'it': 'italian',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tr': 'turkish'
}
    return iso639_to_languages.get(code, "english")

class Normalizer:
    def __init__(self, model: str = "es_core_news_sm"):
        # Pattern for removing digit, symbols or digit+symbol tokens
        self.digitsymbol_pattern = re.compile(r"^[^A-Za-z\s]+$")
        # Load the spacy model and download if not already
        try:
            self.nlp = spacy.load(model)
        except OSError:
            os.system(f"python3 -m spacy download {model}")
            self.nlp = spacy.load(model)

    def ngram_list(self, wordlist: list, n: int = 1):
        # wordlist is a list of words
        # n is the size of the ngram
        # returns a list of ngrams as tuples
        ngrams = []
        for i in range(len(wordlist) - n + 1):
            ngrams.append(" ".join(wordlist[i:i+n]))
        return ngrams

    def normalize(self, doc:str, ngram: int = None, remove_digits=False, remove_stopwords=False, return_pos=False):
        # Create a spacy document object
        doc = self.nlp(doc)
        # Initialize an empty list to store the output
        output = []

        # Use list comprehension to filter and transform tokens
        output = [
            token.pos_ if return_pos else token.lemma_.lower() 
            for token in doc 
            if not (remove_digits and bool(re.match(self.digitsymbol_pattern, token.text))) 
            and not (remove_stopwords and token.is_stop) 
            and token.text not in ["\n", "\r", "\r\r", "\n\n", "\t", "\t\n"]
            ]

        # Return n-gram ist instead of wordlist
        if ngram:
            output = self.ngram_list(output, n=ngram)

        # Return the output list
        return output

def find_outliers(dct: dict):
    """
    Input: Dictionary of document ID: Value.
    The z-score of a value is the number of standard
    deviations it is away from the mean of the data. 
    """
    # Calculate the mean and standard deviation of the list
    values = list(dct.values())
    mean = np.mean(values)
    std = np.std(values)
    # Create an empty list to store the outliers
    outliers = []
    # Loop through the list and append any value that is an outlier to the outliers list
    for key in dct.keys():
        # Calculate the z-score of each value
        z_score = (dct[key] - mean) / std
        # Check if the z-score is greater than 3 or less than -3
        if z_score > 2 or z_score < -2:
            outliers.append(str(key))
    # Return the outliers list
    return outliers

def sample_corpus_perc(corpus: list, sample: int, remove_outliers: bool = False):
    """
    Input: List of dictionaries containing a "text" key. 
    """
    subcorpus = random.sample(corpus, int(len(corpus)*(sample/100)))
    if remove_outliers:
        outliers = find_outliers({doc['id']:len(doc['text']) for doc in corpus})
        for i, doc in enumerate(subcorpus):
            if str(doc["id"]) in outliers:
                subcorpus.pop(i)
    else:
        outliers = []
    print(f"[INFO] Returned random corpus sample of {sample}% with {len(subcorpus)} instances after removing {len(outliers)} outliers.")
    return subcorpus

def sample_corpus_numr(corpus: list, sample: int, remove_outliers: bool = False):
    """
    Input: List of dictionaries containing a "text" key. 
    """
    try:
        subcorpus = random.sample(corpus, sample)
    except ValueError:
        subcorpus = corpus
    if remove_outliers:
        outliers = find_outliers({doc['id']:len(doc['text']) for doc in corpus})
        for i, doc in enumerate(subcorpus):
            if str(doc["id"]) in outliers:
                subcorpus.pop(i)
    else:
        outliers = []
    print(f"[INFO] Returned random corpus sample of {len(subcorpus)} instances after removing {len(outliers)} outliers.")
    return subcorpus

def get_entropy_info(corpus: list, ngram: int = None, pos: bool = False, return_highest_lowest: bool = False):
    """
    Input: List of dictionaries containing a "text" key. 
    """
    normalizer = Normalizer()

    for document in corpus:
        # I do not remove stopwords because then a structured text could have the same perplexity as a random list of words
        normalized_wordlist = normalizer.normalize(doc=document["text"], return_pos=pos, remove_digits=True, remove_stopwords=False, ngram=ngram)
        document["entropy"] = calculate_entropy(normalized_wordlist)
    if return_highest_lowest:
        high_ent = max(corpus, key=lambda x: x["entropy"])
        low_ent = min(corpus, key=lambda x: x["entropy"])
        print(f"Document with highest entropy ({high_ent['entropy']}):\n'''\n{high_ent['text']}'''\n\nDocument with lowest entropy ({low_ent['entropy']}):\n'''\n{low_ent['text']}'''")
    return corpus

def calculate_entropy(word_list: list):
    """
    Complexity or entropy is a measure of how predictable or 
    orderly a sequence of symbols is. A random sequence of 
    symbols has high complexity or entropy, while a structured 
    sequence of symbols has low complexity or entropy.
    """
    # Create a frequency distribution of words in the sample
    freqdist = nltk.FreqDist(word_list)
    # Get the total number of tokens in the distribution
    N = freqdist.N()
    # Initialize the entropy variable
    H = 0
    # Loop through each token and its frequency
    for token, freq in freqdist.items():
        # Compute the probability of the token
        p = freq / N
        # Compute the contribution of the token to the entropy
        H += -p * log2(p)
    # Return the entropy value
    return H


def check_language_and_download_stopwords(text):
    """
    Checks language(s) for a given text and downloads
    stopwords from NLTK into a list.
    """
    nltk.download('stopwords')
    # Initialize list
    stopword_list = []
    # Detecting the language of the text using langdetect
    iso_languages = detect_langs(text)
    # Loop over languages
    for iso in iso_languages:
        language = iso_to_english(iso.lang)
        # Checking if the language is in the list of stopwords languages of nltk
        if language in nltk.corpus.stopwords.fileids():
            # Downloading the stopwords for that language
            stopword_list += set(nltk.corpus.stopwords.words(language))
            # Printing a success message
            print(f"Stopwords for {language} downloaded successfully.")
        else:
            # Printing a failure message
            print(f"Stopwords for {language} not available in nltk.")
    return stopword_list

def raw_wordlist_analyzer(lst):
    """
    Makes TfidfVectorizer use the given raw word list instead of
    trying to tokenize the documents by itself.
    """
    return lst

def cosine_similarity(corpus: list, pos: bool = False, ngram: int = None):
    """
    The vectors compared with cosine can for instance 
    contain frequencies of characters or characters 
    n-grams, hence making it a string similarity measure.
    It assumes a "bag-of-words" vector representation, 
    i.e. compares unordered sets.
    """
    normalizer = Normalizer()
    normalized_corpus = []

    for document in corpus:
        normalized_wordlist = normalizer.normalize(doc=document["text"], remove_digits=True, remove_stopwords=False, return_pos=pos, ngram=ngram)
        normalized_corpus.append(normalized_wordlist)
    
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(analyzer=raw_wordlist_analyzer, smooth_idf=False)
    # Transform the documents into tf-idf vectors
    vectors = vectorizer.fit_transform(normalized_corpus)
    # Compute the cosine similarity matrix
    matrix = np.dot(vectors, vectors.T).toarray()
    # Return the similarity matrix
    return matrix

def cluster_kmeans(corpus: list, pos: bool = False, ngram: int = None): #TODO: Find a way to get a given cluster into a corpus
    normalizer = Normalizer()
    normalized_corpus = []

    for document in corpus:
        normalized_wordlist = normalizer.normalize(doc=document["text"], remove_digits=True, remove_stopwords=False, return_pos=pos, ngram=ngram)
        normalized_corpus.append(normalized_wordlist)

    vectorizer = TfidfVectorizer(analyzer=raw_wordlist_analyzer, smooth_idf=False)
    # Transform the documents into tf-idf vectors
    doc_vectors = vectorizer.fit_transform(normalized_corpus)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2, random_state=13)
    doc_vectors_reduced = pca.fit_transform(doc_vectors.toarray())

    # Cluster documents with K-means
    kmeans = KMeans(n_clusters=4, random_state=13)
    kmeans.fit(doc_vectors_reduced) 
    clusters = kmeans.labels_

    return doc_vectors_reduced, clusters, kmeans

# Sample texts from poetry/ lyrics websites
def sample_poetic_commoncrawl(corpuspath: str, num: int = 100):
    corpus = cc_wet_warc(corpuspath)
    url_query = "lyrics|https://genius.com"
    lyric_corpus = [
        d for d in corpus
        if re.search(url_query, d["uri"]) 
        and "$" not in d["text"]]
    print(f"Lyrical corpus has {len(lyric_corpus)} examples")
    return lyric_corpus

# Sample random texts
def sample_random_commoncrawl(corpuspath: str, num: int):
    corpus = cc_wet_warc(corpuspath)
    sampled_corpus = random.sample(corpus, num)
    print(f"Random corpus has {len(sampled_corpus)} examples")
    return sampled_corpus


def parentheses_per_line_ratio(document, min_threshold = 0.1):
    # Does not work: Not all lyrics use parenthesis. 
    # In fact, most random webpages use parenthesis for web structure
    # Heuristic to check if a document has at least 10% of lines ending in parenthesis
    lines = document.split('\n')
    total_lines = len(lines)
    lines_with_parentheses = sum(1 for line in lines if re.search(r'\(.*?\)\s*$', line))

    if total_lines > 0 and lines_with_parentheses / total_lines > min_threshold:
        return True
    else:
        return False
    

def keyword_search(document):
    # Heuristic to check if a document contains one of the keywords
    query = r"chorus|verse|refrain|song lyrics"

    if re.search(query, document.lower()):
        return True
    else:
        return False


def common_vowels(document, min_threshold: float=0.05, lang='english'):
    # Heuristic to check if most words end with the most common set of vowels
    # Does not work, plus there are lyrics in other languages that do not
    # register 
    def get_last_word(sentence: str):
        if sentence.strip() == "":
            return None
        else:
            last_word = word_tokenize(sentence, lang)[-1]
            return last_word

    def get_most_common_vowel_set(sentences: list):
        vowels = set('aeiou')
        sentence_vowels = []

        for sentence in sentences:
            last_word = get_last_word(sentence)
            if last_word:
                last_word_vowels = set(char for char in last_word.lower() if char in vowels)
                sentence_vowels.append(tuple(last_word_vowels))

        return Counter(sentence_vowels).most_common(1)

    sentences = document.splitlines()

    most_common_vowel_set, most_common_count = get_most_common_vowel_set(sentences)[0]


    words_with_most_common_vowels = sum(
        1 for sentence in sentences
        if get_last_word(sentence) 
        and set(char for char in get_last_word(sentence).lower()) == set(most_common_vowel_set)
        )
    
    # It is lyrical
    if (words_with_most_common_vowels / len(sentences)) >= min_threshold:
        return True 
    else:
        return False

# DONE
# Heuristic to check if a document has a certain avg. word per line
def avg_word_per_line(document, word_count_threshold=7, word_count_upper_limit=9, min_threshold=0.1, lang='english'):
    lines = sent_tokenize(document, lang)
    num_lines_within_threshold = sum(
        1 for line in lines 
        if word_count_threshold <= len(word_tokenize(line, lang)) <= word_count_upper_limit
        )
    if (num_lines_within_threshold / len(lines)) >= min_threshold:
        return True
    else:
        return False


def apply_heuristic(doc_text, title):
    code = detect(doc_text)
    lang = iso_to_english(code)
    if "Chorus" in title:
        return parentheses_per_line_ratio(doc_text)
    if "Rhyme" in title:
        return common_vowels(doc_text, lang=lang)
    if "Length" in title:
        return avg_word_per_line(doc_text, lang=lang)
    if "Vocab" in title:
        return keyword_search(doc_text)


def poetry_confusion_matrix(poetic_sample, non_poetic_sample, title):
    true_positives = sum(1 for doc in poetic_sample if apply_heuristic(doc["text"], title) is True)
    false_positives = len(poetic_sample) - true_positives
    true_negatives = sum(1 for doc in non_poetic_sample if apply_heuristic(doc["text"], title) is False)
    false_negatives = len(non_poetic_sample) - true_negatives

    confusion_matrix = {
        "TP": true_positives, # Correctly identified as lyric
        "FP": false_positives, # Misidentified as lyric
        "TN": true_negatives, # Correctly identifies as non-lyric
        "FN": false_negatives # Misidentified as non-lyric
    }

    return confusion_matrix

def get_most_common_words(strings_list, top_n=10):
    # Combine all the strings into a single text
    combined_text = ' '.join(strings_list)
    
    # Use regex to extract individual words (excluding punctuation)
    words = re.findall(r'\b\w+\b', combined_text.lower())
    words = nltk.ngrams(words, n=2)
    
    # Count the occurrences of each word
    word_counts = Counter(words)
    
    # Get the top N most common words
    most_common_words = word_counts.most_common(top_n)
    
    return most_common_words