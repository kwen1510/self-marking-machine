import streamlit as st

# Library for Sentence Similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Library for Entailment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Library for keyword extraction
import yake


# Load models and tokenisers for both sentence transformers and text classification

sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

text_classification_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")



### Streamlit interface ###
      
st.title("Sentence Similarity")

sidebar_selectbox = st.sidebar.selectbox(
    "What would you like to work with?",
    ("Compare two sentences", "Bulk upload and mark")
)

# Streamlit form elements (default to "Compare two sentences")

if sidebar_selectbox == "Compare two sentences":

       st.subheader("Compare the similarity between two sentences")
       
       with st.form("submission_form", clear_on_submit=False):
       
              sentence_1 = st.text_input("Sentence 1 input")
              
              sentence_2 = st.text_input("Sentence 2 input")
              
              submit_button_compare = st.form_submit_button("Compare Sentences")
              
       # If submit_button_compare clicked
       if submit_button_compare:

              print("Comparing sentences...")

              ### Compare Sentence Similarity ###
       
              # Perform calculations
              
              #Initialise sentences
              sentences = []
              
              # Append input sentences to 'sentences' list
              sentences.append(sentence_1)
              sentences.append(sentence_2)
              
              # Create embeddings for both sentences
              sentence_embeddings = sentence_transformer_model.encode(sentences)
              
              cos_sim = cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[1].reshape(1, -1))[0][0]
              cos_sim = round(cos_sim * 100) # Convert to percentage and round-off
             
                     
              # st.write('Similarity between "{}" and "{}" is {}%'.format(sentence_1,
              #        sentence_2, cos_sim))

              st.subheader("Similarity")
              st.write(f"Similarity between the two sentences is {cos_sim}%.")


              ### Text classification - entailment, neutral or contradiction ###

              raw_inputs = [f"{sentence_1}</s></s>{sentence_2}"]

              inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

              # print(inputs)

              outputs = text_classification_model(**inputs)

              outputs = torch.nn.functional.softmax(outputs.logits, dim = -1)
              # print(outputs)

              # argmax_index = torch.argmax(outputs).item()

              print(text_classification_model.config.id2label[0], ":", round(outputs[0][0].item()*100,2),"%")
              print(text_classification_model.config.id2label[1], ":", round(outputs[0][1].item()*100,2),"%")
              print(text_classification_model.config.id2label[2], ":", round(outputs[0][2].item()*100,2),"%")

              st.subheader("Text classification for both sentences:")

              st.write(text_classification_model.config.id2label[1], ":", round(outputs[0][1].item()*100,2),"%")
              st.write(text_classification_model.config.id2label[0], ":", round(outputs[0][0].item()*100,2),"%")
              st.write(text_classification_model.config.id2label[2], ":", round(outputs[0][2].item()*100,2),"%")


              ### Extract keywords with YAKE ### (might make more sense with word cloud)

              st.subheader("Keywords:")

              kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
              keywords = kw_extractor.extract_keywords(sentence_2)

              # keywords_array = []

              for kw, v in keywords:
                # print("Keyphrase: ", kw, ": score", v)
                # keywords_array.append(kw)

                st.write(kw)





if sidebar_selectbox == "Bulk upload and mark":

       st.subheader("Bulk compare similarity of sentences")
       
       sentence_reference = st.text_input("Reference sentence input")
       
       # Only allow user to upload CSV files
       data_file = st.file_uploader("Upload CSV",type=["csv"])
       
       if data_file is not None:
              with st.spinner('Wait for it...'):
                     file_details = {"filename":data_file.name, "filetype":data_file.type, "filesize":data_file.size}
                     # st.write(file_details)
                     df = pd.read_csv(data_file)
                     
                     # Get length of df.shape (might not need this)
                     #total_rows = df.shape[0]
                     
                     similarity_scores = []
                     
                     for idx, row in df.iterrows():
                            # st.write(idx, row['Sentences'])
                            
                            # Create an empty sentence list
                            sentences = []
                            
                            # Compare the setences two by two
                            sentence_comparison = row['Sentences']
                            sentences.append(sentence_reference)
                            sentences.append(sentence_comparison)
                            
                            sentence_embeddings = sentence_transformer_model.encode(sentences)
                            
                            cos_sim = cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[1].reshape(1, -1))[0][0]
                            cos_sim = round(cos_sim * 100)
                            
                            similarity_scores.append(cos_sim)                    
                     
                     # Append new column to dataframe
                     
                     df['Similarity (%)'] = similarity_scores
                     
                     st.dataframe(df)
              st.success('Done!')  
              
              @st.cache
              def convert_df(df):
                     return df.to_csv().encode('utf-8')
                     
              csv = convert_df(df)
              
              st.download_button(
                 "Press to Download",
                 csv,
                 "marked assignment.csv",
                 "text/csv",
                 key='download-csv'
              )