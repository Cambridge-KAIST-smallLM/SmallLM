#SimLex999
import pandas as pd
df = pd.read_csv(open("SimLex999.txt"), delimiter = "\t")
word1 = list(df.word1)
word2 = list(df.word2)
gold_scores = list(df.SimLex999) #gold scores are on a scale rated by human annotators

#Rare Words (Card600)
df = pd.read_csv(open("Card600/card600.csv"), delimiter = ',')
df.columns = ['word1', 'word2', 'card600']
word1 = list(df.word1)
word2 = list(df.word2)
gold_scores = list(df.card600)

#SimVerb3500 – this has relations like synonym, antonym, cohyponym which are good to report. Scale 0 - 10. (https://aclanthology.org/D16-1235.pdf) 
df = pd.read_csv(open("/content/drive/MyDrive/LTL/simverb3500/simverb3500.txt"), delimiter = "\t")
df.columns = ['word1', 'word2', 'V', 'SimVerb3500', 'Relation']
word1 = list(df.word1)
word2 = list(df.word2)
gold_scores = list(df.SimVerb3500)

#Some other datasets, for completeness in Appendix – RG65 and WordSim353 [older datasets - less coverage]
df = pd.read_csv(open("rg65.txt"), delimiter = ";")
df.columns = ['word1', 'word2', 'rg65']
word1 = list(df.word1)
word2 = list(df.word2)
gold_scores = list(df.rg65)


#FOR ALL DATASETS: example function to process word embeddings using mean-pooling
string_features1, string_features2 = [], []
def get_embedding(words):
    embeddings = []
    for i in tqdm(range(0, len(words), 128)):
        toks = tokenizer(
            words[i : i + 128], 
            max_length=64, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**toks)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        mean_pooled = last_hidden_state.mean(dim=1).cpu().detach().numpy()  # Mean over sequence
        embeddings.append(mean_pooled)
    return np.concatenate(embeddings, axis=0)
string_features1_stacked = get_embedding(word1) # Get embeddings for both word lists
string_features2_stacked = get_embedding(word2)
sims = [
    1 - spatial.distance.cosine(string_features1_stacked[i], string_features2_stacked[i])
    for i in range(len(string_features1_stacked))
] # Compute cosine similarity

# RESULTS TO REPORT - Compute Spearman correlation aggregated across all word lists. 
score = spearmanr(gold_scores, sims)[0]
print("Spearman correlation:", score)

#For SimLex: How to report scores by POS
noun = df[df["POS"]=="N"]
verb = df[df["POS"]=="V"]
adj = df[df["POS"]=="A"]
gold_scores = list(noun["SimLex999"])
sims = list(noun["BERT_Score"])
from scipy.stats.stats import pearsonr,spearmanr
score = spearmanr(gold_scores, sims)[0]
print(score)

