#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[15]:


# Specify the directory you want to change to
new_directory = "C:/Users/ray/Desktop/pyPapers"  # Replace with your desired path

# Change to the new directory
os.chdir(new_directory)

# Verify the change
print("Current Working Directory:", os.getcwd())


# In[19]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os


# In[20]:


# Define the file path
file_path = 'multimodalDataV5.xlsx'


# In[21]:


# Load the Excel file from the current working directory
df = pd.read_excel(file_path)


# In[22]:


# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder()


# In[23]:


# Ordinal Encoding for 'severity'
df['severity_encoded'] = ordinal_encoder.fit_transform(df[['severity']])


# In[24]:


# Ordinal Encoding for 'storyPoint'
df['storyPoint_encoded'] = ordinal_encoder.fit_transform(df[['storyPoint']])


# In[26]:


# Save the result to the same working directory
output_path = 'encoded_data.xlsx'
df.to_excel(output_path, index=False)


# In[27]:


print(f"Encoded data saved to {output_path}")


# In[ ]:





# In[47]:


import numpy as np
import pandas as pd

# Load the encoded data
df_encoded = pd.read_excel('encoded_data.xlsx')


# In[49]:


print("df_encoded")
print(df_encoded)


# In[50]:


# Convert the embedded data columns to numpy arrays
story_embeddings = np.array(df_encoded['story_embedding'].tolist())
image_feature_embeddings = np.array(df_encoded['imageFeature_embedding'].tolist())


# In[51]:


# Ensure embeddings are 2D
if story_embeddings.ndim == 1:
    story_embeddings = story_embeddings.reshape(-1, 1)
if image_feature_embeddings.ndim == 1:
    image_feature_embeddings = image_feature_embeddings.reshape(-1, 1)


# In[52]:


# Extract and reshape encoded categorical data
severity_encoded = df_encoded['severity_encoded'].values.reshape(-1, 1)
story_point_encoded = df_encoded['storyPoint_encoded'].values.reshape(-1, 1)


# In[53]:


# Print shapes for debugging
print("Story embeddings shape:", story_embeddings.shape)
print("Image feature embeddings shape:", image_feature_embeddings.shape)
print("Severity encoded shape:", severity_encoded.shape)
print("Story point encoded shape:", story_point_encoded.shape)


# In[54]:


# Ensure all arrays have the same number of rows
num_rows = min(story_embeddings.shape[0], image_feature_embeddings.shape[0], severity_encoded.shape[0], story_point_encoded.shape[0])
print("Number of rows to match:", num_rows)


# In[55]:


# Adjust arrays to have the same number of rows
story_embeddings = story_embeddings[:num_rows]
image_feature_embeddings = image_feature_embeddings[:num_rows]
severity_encoded = severity_encoded[:num_rows]
story_point_encoded = story_point_encoded[:num_rows]


# In[56]:


# Combine the features into a single array
combined_features = np.concatenate(
    (story_embeddings, image_feature_embeddings, severity_encoded, story_point_encoded), axis=1
)

print("Combined features shape:")
print(combined_features.shape)


# In[57]:


# Convert combined features to a DataFrame
combined_df = pd.DataFrame(combined_features, columns=[
    'Story_Embedding', 'Image_Feature_Embedding', 'Severity_Encoded', 'StoryPoint_Encoded'
])

# Define output file path
output_file_path = 'combined_features.xlsx'

# Save the DataFrame to an Excel file
combined_df.to_excel(output_file_path, index=False)

print(f"Combined features saved to {output_file_path}")

