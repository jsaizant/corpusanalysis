from read import *
from tools import *
from visul import *
import json

### Entropy histogram

# Get one corpus
#en_cc = cc_wet_warc("data/CC-MAIN-202201.warc.wet.gz", "eng")
#es_wiki = wikipedia("data/wikipedia_es_20230525/")

# Sample corpus
#es_wiki = sample_corpus_numr(es_wiki, sample=1000, remove_outliers=False)

# Write corpus
# with open("data/cc-es.jsonl", "w") as fout:
#     fout.write(json.dumps(subcorpus, indent=1))
# with open("data/wiki-es.jsonl", "w") as fout:
#     fout.write(json.dumps(es_wiki, indent=1))

# Load corpus
# with open("data/cc-es.jsonl", "r") as fin:
#     es_cc = json.load(fin)
# with open("data/wiki-es.jsonl", "r") as fin:
#     es_wiki = json.load(fin)
# # Sample corpus
# es_wiki = sample_corpus_numr(es_wiki, sample=500, remove_outliers=False)
# es_cc = sample_corpus_numr(es_cc, sample=500, remove_outliers=False)

# Calculate entropy per document
#subcorpus = get_entropy_info(subcorpus, return_highest_lowest=False)

# Visualize one histogram
#entropy_histogram(subcorpus)

### Multiple corpus entropy comparison with density curves
#entropy_density_curves_corpora([es_cc, es_wiki], pos=True, ngram=3)

### Corpora cosine similarity
#cosine_similarity_corpora([es_cc, es_wiki], pos=False)

### Corpora cosine similarity
#document_clustering_corpora([es_cc, es_wiki], pos=True, ngram=3)



# Get poetry path
lyricpath = os.path.join(os.getcwd(), "data/commoncrawl_lyrical_webs.jsonl")
randompath = os.path.join(os.getcwd(), "data/commoncrawl_random_webs.jsonl")
if os.path.exists(lyricpath):
    # Load corpus
    with open(lyricpath, "r") as fin:
        poetry_sample = json.load(fin)
        print(f"Lyrical corpus has {len(poetry_sample)} examples")
else:
    # Sample poetry and random corpora
    warc_list = os.listdir("./data/")
    poetry_sample = []
    for file in warc_list:
        filepath = os.path.join(os.getcwd(), "data/", file )
        sample = sample_poetic_commoncrawl(filepath)
        if sample: 
            poetry_sample.extend(sample)
        else:
            print("no files")
    with open(lyricpath, "w") as fout:
        fout.write(json.dumps(poetry_sample))

n = 200

filepath = os.path.join(os.getcwd(), "data/", os.listdir("./data/")[1])
random_sample = sample_random_commoncrawl(filepath, n)

# Write random sample
with open(randompath, "w") as fout:
    fout.write(json.dumps(random_sample))

# See if heuristic to detect poetry works
titles = ["(Chorus) More than 10 percent of lines have parenthesis at the end",
          "(Vocab) Document contains related keywords", 
          "(Rhyme) More than 5 percent of final sentence words have a common set of vowels", 
          "(Length) More than 40 percent of lines have between 7 and 9 words"]

for t in titles:
    cf = poetry_confusion_matrix(poetry_sample, random_sample, t)
    print(cf)
    plot_confusion_matrix(cf, title=t)