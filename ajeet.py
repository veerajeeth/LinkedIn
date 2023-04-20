# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:48:53 2023

@author: shubh
"""
import streamlit as st
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string  





# Load data
link = pd.read_csv(r"C:/Users/shubh/linkedin_data 2023-03-31_final.csv")
print(link.columns)
st.write(link.head())

link.drop(columns=['Date'],axis=1,inplace=True)

link[['City', 'state']] = link['Loaction'].str.split(', ',n=1, expand=True)
link.drop(columns=['Loaction'],axis=1,inplace=True)


# define a function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


#remove all the punctuations
link1 = link.copy()
link1.loc[:, 'Job_Name']  =link1.loc[:, 'Job_Name'] .str.replace('[{}]'.format(string.punctuation), '', regex=True)
link1.loc[:, 'Followers']  =link1.loc[:, 'Followers'] .str.replace('[{}]'.format(string.punctuation), '', regex=True)
link1.loc[:, 'state']  =link1.loc[:, 'state'] .str.replace('[{}]'.format(string.punctuation), '', regex=True)

# Remove the word "followers" from the "Followers" column
link1['Followers'] = link1['Followers'].str.replace('followers', '')


# Remove the word "Applicant" from the "Applicant" column
link1['Applicant'] = link['Applicant'].str.replace('applicants', '')


# Remove the word "India" from the "state" column
link1['state'] = link1['state'].str.replace('india', '')


#checking for missing value in any column
link1.isnull().sum()


#Replacing NULL values in catogrical Columns using Mode
mode1=link1['Applicant'].mode().values[0]
mode2=link1['Industry'].mode().values[0]
mode3=link1['state'].mode().values[0]
link1['Applicant']=link1['Applicant'].replace(np.nan,mode1)
link1['Industry']=link1['Industry'].replace(np.nan,mode2)
link1['state']=link1['state'].replace(np.nan,mode3)




link1.isnull().sum()



link1 = link1.drop_duplicates()





link1['City'].value_counts().head(10).to_frame().style.set_caption('Most Demand Company').background_gradient(cmap='Blues')



link1['Company'].value_counts().head(10).to_frame().style.set_caption('Most Demand Company').background_gradient(cmap='Blues')


link1['Job_Name'].value_counts().head(10).to_frame().style.set_caption('Most Demand Company').background_gradient(cmap='Blues')




import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 




link1['Job_Name'].value_counts()[:5].plot(kind = 'pie',autopct='%1.1f%%',startangle=180,shadow=True ,explode = [0.1,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
st.pyplot(fig)  




link1['Company'].value_counts()[:5].plot(kind = 'pie',autopct='%1.1f%%',startangle=180,shadow=True ,explode = [0.1,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
st.pyplot(fig) 




link1['City'].value_counts()[:5].plot(kind = 'pie',autopct='%1.1f%%',startangle=180,shadow=True ,explode = [0.1,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
st.pyplot(fig) 




link1['Invovlement'].value_counts()[:5].plot(kind = 'pie',autopct='%1.1f%%',startangle=180,shadow=True ,explode = [0.1,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
st.pyplot(fig) 




link1['Industry'].value_counts()[:5].plot(kind = 'pie',autopct='%1.1f%%',startangle=180,shadow=True ,explode = [0.1,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
st.pyplot(fig) 





fig, ax = plt.subplots(nrows=1, ncols=1)
a = sns.barplot(x=link1["Company"].value_counts().sort_values(ascending=False).head(10).index, y=link1["Company"].value_counts().sort_values(ascending=False).head(10), ax=ax)

sns.despine(bottom=False, left=False)

spots = link1["Company"].value_counts().sort_values(ascending=False).index[0:10]
for p in a.patches:
    a.annotate('{:.2f}%'.format((p.get_height() / 742) * 100), (p.get_x() + 0.2, p.get_height() + 1), fontsize=12)

plt.title('\n Top 10 Companies with Maximum Number of Job Postings \n', size=16, color='black')
plt.xticks(fontsize=13, rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('\n Company name \n', fontsize=13, color='black')
plt.ylabel('\n Count \n', fontsize=13, color='black')

st.pyplot(fig)




#importing libraries for content-based recommendation.
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings


link1_copy=link1



link1_copy = link1.reset_index(drop=True).copy()


link1_copy.loc[:, 'Job_Name'] = link1_copy['Job_Name'].fillna('')
link1_copy.loc[:, 'City'] = link1_copy['City'].fillna('')
link1_copy.loc[:, 'Description'] = link1_copy['Job_Name'] + link1_copy['City'] 



from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=0,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            stop_words = 'english')



# Fitting the TF-IDF on the 'Job_name' text
tfv_matrix = tfv.fit_transform(link1_copy['Description'])


tfv_matrix




tfv_matrix.shape 


#find the most similar words 
cosine_sim = linear_kernel(tfv_matrix, tfv_matrix)




from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)




sig[0]



#  mapping of indices and job_name
indices = pd.Series(link1_copy.index, index=link1_copy['Job_Name']).drop_duplicates()






indices


def get_top_similar_jobs(Job_Name):
    # Get the indices of jobs matching the given Job_Name
    Job_indices = link1_copy.index[link1_copy['Job_Name'] == Job_Name].tolist()
    
    # Get the cosine similarity scores between the given Job_Name and all other Job_Names
    Job_sim_scores = cosine_sim[Job_indices].mean(axis=0)
    
    # Get the indices of the top 5 jobs with the highest similarity scores
    top_job_indices = Job_sim_scores.argsort()[::-1][:5]
    
    # Get the job names and corresponding similarity scores
    top_jobs = [(idx, Job_sim_scores[idx]) for idx in top_job_indices]
    
    # Create a list of top similar jobs
    top_jobs = [(link1_copy.loc[idx, 'Job_Name'], link1_copy.loc[idx, 'Company'], link1_copy.loc[idx, 'Followers'], score) for idx, score in top_jobs]
    
    # Create a pandas dataframe from the list of top similar jobs
    df = pd.DataFrame(top_jobs, columns=['Job_Name', 'Company', 'Followers', 'Similarity_Score'])
    
    # Return the top 5 similar jobs for the given Job_Name
    return df.head(5)






# Define the sidebar
st.sidebar.header('Job Recommendation')
job_name = st.sidebar.selectbox('Select a job:', link1_copy['Job_Name'].unique())

# Define the main content
st.title('Top 5 Similar Jobs')
if job_name:
    top_jobs = get_top_similar_jobs(job_name)
    st.table(top_jobs[['Job_Name', 'Company', 'Followers', 'Similarity_Score']])
else:
    st.warning('Please select a job from the sidebar.')







  




job_data =link1_copy




job_data.drop(columns=['Description'],axis=1,inplace=True)




# Create a TfidfVectorizer object to convert the job name and city into vectors
tfidf = TfidfVectorizer(stop_words='english')

# Create a matrix of job name and city vectors
job_city_matrix = tfidf.fit_transform(job_data['Job_Name'] + ' ' +job_data['City'])

# Compute the cosine similarity of job name and city vectors
cosine_sim = cosine_similarity(job_city_matrix, job_city_matrix)

# Define a function to get the top 5 similar job names based on the given city
def get_top_similar_jobs(city_name):
    # Get the indices of jobs matching the given city
    city_indices = job_data.index[job_data['City'] == city_name].tolist()
    
    # Get the cosine similarity scores between the given city and all other cities
    city_sim_scores = cosine_sim[city_indices].mean(axis=0)
    
    # Get the indices of the top 5 jobs with the highest similarity scores
    top_job_indices = city_sim_scores.argsort()[::-1][:5]
    
    # Get the job names and corresponding similarity scores
    top_jobs = [(idx, city_sim_scores[idx]) for idx in top_job_indices]
    
    # Create a list of top similar jobs
    top_jobs = [(job_data.loc[idx, 'Job_Name'], job_data.loc[idx, 'Company'], job_data.loc[idx, 'Followers'], score) for idx, score in top_jobs]
    
    # Create a pandas dataframe from the list of top similar jobs
    df = pd.DataFrame(top_jobs, columns=['Job_Name', 'Company', 'Followers', 'Similarity_Score'])
    
    # Return the top 5 similar jobs for the given city
    return df.head(5)




# Create the Streamlit app interface
st.title('Job Recommendation System')

# Add a dropdown to select the city
city_name = st.selectbox('Select a city:', job_data['City'].unique())

# Add a button to get the top similar jobs for the selected city
if st.button('Get Top Similar Jobs'):
    # Call the function to get the top similar jobs for the selected city
    top_similar_jobs = get_top_similar_jobs(city_name)
    
    # Display the top similar jobs
    st.write(f'Top 5 similar jobs in {city_name}:')
    st.table(top_similar_jobs)
    
    
    











