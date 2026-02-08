#for dataframe manipulation
import numpy as np 
import pandas as pd
#regular expressoin toolkit
import re
#NLP toolkits
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.tokenize import word_tokenize
#for downloading BERT
# !pip install sentence_transformers
from sentence_transformers import SentenceTransformer
#for finding most similar text vectors
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
# clean text for NLP processing
    # Lowercase everything 
    text = str(text).lower()
    # Remove URLs 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    # Remove punctuation 
    text = re.sub(r'[^\w\s]', ' ', text)
    # Tokenize into words (splits any leftover spaces/punctuation)
    tokens = word_tokenize(text)
    # Join tokens back into a clean string with single spaces
    cleaned_text = ' '.join(tokens)  

    return text

# Training set and testing set requirements: 
# - 1st column 'desription' contains transaction description 
# - 2nd colum 'class' contains the label for each transaction
# Load training set 
df_transaction_description_train = pd.read_csv("TrainingMint_Export_201401_202312.csv")
# De-duplicate training embeddings 
df_transaction_description_train = (
    df_transaction_description_train
    .drop_duplicates(subset=["description", "class"])
    .reset_index(drop=True)
)
# Load testing set 
df_transaction_description_test = pd.read_csv('merged_statements.csv')

text_raw_train = df_transaction_description_train['description']  
text_train = text_raw_train.apply(lambda x: clean_text(x))
# print(text_BERT.head())

######################################
### Download pre-trained BERT model###
######################################

# This may take some time to download and run 
# depending on the size of the input

embedding_input_train = text_train.tolist()
model = SentenceTransformer('paraphrase-mpnet-base-v2') 
embeddings = model.encode(embedding_input_train, show_progress_bar = True)
embedding_train = np.array(embeddings)

# Load texts
text_test_raw = df_transaction_description_test['description']

# Apply data cleaning function to testing data
text_test = text_test_raw.apply(lambda x: clean_text(x))
# print(text_test_BERT.head())

# Apply BERT embedding to testing data
embedding_input_test = text_test.tolist()
embeddings_test = model.encode(embedding_input_test, show_progress_bar = True)
embedding_test = np.array(embeddings_test)

df_embedding_bert_test = pd.DataFrame(embeddings_test)

# Find the most similar word embedding with unseen data in the training data using cosine distance
similarity = cosine_similarity(embedding_test, embedding_train)

TOP_K = 5
SIMILARITY_THRESHOLD = 0.70

# Top-K indices per test transaction
top_k_idx = np.argsort(similarity, axis=1)[:, -TOP_K:]

matched_description = []
matched_class = []
matched_score = []

for i, idxs in enumerate(top_k_idx):
    scores = similarity[i, idxs]

    # Filter by threshold
    valid = [(idx, score) for idx, score in zip(idxs, scores) if score >= SIMILARITY_THRESHOLD]

    if not valid:
        matched_description.append(None)
        matched_class.append("Uncategorized")
        matched_score.append(scores.max())
        continue

    # Majority vote on class
    classes = df_transaction_description_train.iloc[[idx for idx, _ in valid]]['class']
    most_common_class = classes.value_counts().idxmax()

    # Keep best matching description for inspection
    best_idx = max(valid, key=lambda x: x[1])[0]

    matched_description.append(df_transaction_description_train.iloc[best_idx]['description'])
    matched_class.append(most_common_class)
    matched_score.append(scores.max())

# Return dataframe for most similar embedding/transactions in training dataframe
# data_inspect = df_transaction_description.iloc[index_similarity, :].reset_index(drop = True)

# unseen_verbatim = text_test_raw
# matched_verbatim = data_inspect['description']
# annotation = data_inspect['class']

d_output = {
            'unseen_transaction': text_test_raw,
            'matched_transaction': matched_description, 
            'matched_class': matched_class,
            'similarity_score': matched_score
            }

# convert output to df 
df_output = pd.DataFrame.from_dict(d_output)
# df_output.to_csv('classified_output.csv')

# merge the class colum to the transaction df 
merged_df = pd.concat([df_transaction_description_test['date'],
                       df_transaction_description_test['description'],
                       df_output['matched_class'], 
                       df_transaction_description_test['amount'], 
                       df_transaction_description_test['source'], 
                       df_transaction_description_test['balance']], axis=1)
print(merged_df.head())
merged_df.to_csv('transactions_classfied.csv', index=False)
print('transactions_classfied.csv exported\nend')