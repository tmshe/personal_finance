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

###############################################
### Define a function for NLP data cleaning ###
###############################################

def clean_text_BERT(text):

    # Convert words to lower case.
    text = text.lower()

    # Remove special characters and numbers. This also removes the dates 
    # which are not important in classifying expenses
    text = re.sub(r'[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+', '', str(text).strip())
  
    # Tokenise 
    text_list = word_tokenize(text)
    result = ' '.join(text_list)
    return result

# Training set and testing set requirements: 
# - 1st column 'desription' contains transaction description 
# - 2nd colum 'class' contains the label for each transaction
# Load training set 
df_transaction_description = pd.read_csv("TrainingMint_Export_201401_202312.csv")
# Load testing set 
df_transaction_description_test = pd.read_csv('merged_statements.csv')

text_raw = df_transaction_description['description']  
text_BERT = text_raw.apply(lambda x: clean_text_BERT(x))
# print(text_BERT.head())

######################################
### Download pre-trained BERT model###
######################################

# This may take some time to download and run 
# depending on the size of the input

bert_input = text_BERT.tolist()
model = SentenceTransformer('paraphrase-mpnet-base-v2') 
embeddings = model.encode(bert_input, show_progress_bar = True)
embedding_BERT = np.array(embeddings)

# Load texts
text_test_raw = df_transaction_description_test['description']

# Apply data cleaning function to testing data
text_test_BERT = text_test_raw.apply(lambda x: clean_text_BERT(x))
# print(text_test_BERT.head())

# Apply BERT embedding to testing data
bert_input_test = text_test_BERT.tolist()
embeddings_test = model.encode(bert_input_test, show_progress_bar = True)
embedding_BERT_test = np.array(embeddings_test)

df_embedding_bert_test = pd.DataFrame(embeddings_test)

# Find the most similar word embedding with unseen data in the training data using cosine distance
similarity_new_data = cosine_similarity(embedding_BERT_test, embedding_BERT)
similarity_df = pd.DataFrame(similarity_new_data)

# Returns index for most similar embedding
# See first column of the output dataframe below
index_similarity = similarity_df.idxmax(axis = 1)

# Return dataframe for most similar embedding/transactions in training dataframe
data_inspect = df_transaction_description.iloc[index_similarity, :].reset_index(drop = True)

unseen_verbatim = text_test_raw
matched_verbatim = data_inspect['description']
annotation = data_inspect['class']

d_output = {
            'unseen_transaction': unseen_verbatim,
            'matched_transaction': matched_verbatim, 
            'matched_class': annotation
            
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