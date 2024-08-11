#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[16]:


# Specify the directory you want to change to
new_directory = "C:/Users/ray/Desktop/pyPapers"  # Replace with your desired path

# Change to the new directory
os.chdir(new_directory)

# Verify the change
print("Current Working Directory:", os.getcwd())


# In[17]:


get_ipython().system('pip install pandas transformers openpyxl torch')


# In[18]:


import os
import pandas as pd

# Get the current working directory
cwd = os.getcwd()


# In[31]:


# List all files in the current working directory
print("Files in the current directory:", os.listdir(os.getcwd()))


# In[32]:


# Specify the file name
input_file_name = 'multimodalDataV4.xlsx'  # Replace with your actual file name
input_file_path = os.path.join(cwd, input_file_name)

# Load the Excel file
df = pd.read_excel(input_file_path)

# Display the first few rows to confirm the data is loaded correctly
print(df.head())


# In[27]:


from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# In[33]:


# Tokenize the 'story' and 'imageFeature' columns
def tokenize_text(text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

df['tokenized_story'] = df['story'].apply(tokenize_text)
df['tokenized_imageFeature'] = df['imageFeature'].apply(tokenize_text)

# Display tokenized data
print(df['tokenized_story'].head())
print(df['tokenized_imageFeature'].head())


# In[34]:


# To inspect the actual tokenized data, we can access each of these components separately:

# Example for the first row
first_tokenized_story = df['tokenized_story'].iloc[0]

# Access input_ids
print(first_tokenized_story['input_ids'])

# Access attention_mask
print(first_tokenized_story['attention_mask'])

# Access token_type_ids (usually not as relevant unless doing specific tasks)
print(first_tokenized_story['token_type_ids'])


# In[37]:


# Step 5: Generate BERT embeddings for the tokenized text in both columns
def get_bert_embeddings(tokenized_text):
    with torch.no_grad():
        outputs = model(**tokenized_text)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings

df['story_embedding'] = df['tokenized_story'].apply(get_bert_embeddings)
df['imageFeature_embedding'] = df['tokenized_imageFeature'].apply(get_bert_embeddings)
print(df['story_embedding'].head())
print(df['imageFeature_embedding'].head())


# In[43]:


# Convert PyTorch tensor to NumPy array and flatten it
df['story_embedding'] = df['story_embedding'].apply(lambda x: x.numpy().flatten() if torch.is_tensor(x) else x.flatten())
df['imageFeature_embedding'] = df['imageFeature_embedding'].apply(lambda x: x.numpy().flatten() if torch.is_tensor(x) else x.flatten())

# Display the first few rows of the processed embeddings
print(df['story_embedding'].head())
print(df['imageFeature_embedding'].head())


# In[44]:


# Step 7: Save the embeddings to an Excel file in the current working directory
output_file_name = 'output_file.xlsx'  # Replace with your desired file name
output_file_path = os.path.join(cwd, output_file_name)
df.to_excel(output_file_path, index=False)
print(f"Output file saved to: {output_file_path}")

