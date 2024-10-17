import PyPDF2
from PyPDF2 import PdfReader
import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import fitz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

import numpy as np


##Preprocessing the PDF:
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Example usage
pdf_path = "c:/Users/shrek/OneDrive/Documents/GitHub/Springboard/Springboard/NGOTextFile.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
#print(extracted_text) 

def segment_pdf(file_path, break_pages):
    segments = []
    current_segment = []

    with fitz.open(file_path) as doc:
        for page_number in range(len(doc)):
            # Check if this page is a break page
            if page_number + 1 in break_pages:
                # Add the current segment to segments (if it's not empty)
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []

            # Extract text from the current page
            page = doc.load_page(page_number)
            page_text = page.get_text()
            current_segment.append(page_text)

        # Add the last segment
        if current_segment:
            segments.append(' '.join(current_segment))

    return segments

# List of pages where each new document starts
break_pages = [7, 11, 15, 18, 24, 27, 31,  45, 49, 53, 57, 60, 65, 70, 76, 80, 85, 90, 98]  # Modify this list as needed
# List of regions corresponding to each document
regions = [
    "Global",  # Document 1
    "Global",  # Document 2
    "Africa",  # Document 3
    "Latin America",  # Document 4
    "Asia",  # Document 5
    "Asia",  # Document 6
    "Global",  # Document 7
    "Latin America",  # Document 8
    "Global",  # Document 9
    "Global",  # Document 10
    "Global",  # Document 11
    "Global",  # Document 12
    "Global",  # Document 13
    "Asia",  # Document 14
    "Latin America",  # Document 15
    "Latin America",  # Document 16
    "Asia",  # Document 17
    "Global",  # Document 18
    "Middle East",  # Document 19
    "Africa"   # Document 20
]

# Segment the PDF
pdf_segments = segment_pdf(pdf_path, break_pages)


# Associate each document (segment) with its region
document_data = [(segment, region) for segment, region in zip(pdf_segments, regions)]

# Displaying the first few characters of each segment to verify
segment_previews = {i+1: segment[:500] + '...' for i, segment in enumerate(pdf_segments)}

def preprocess_segments(segments):
    preprocessed_segments = []
    for segment in segments:
        # Tokenization
        from nltk.tokenize import word_tokenize
#print(extracted_text) 

def segment_pdf(file_path, break_pages):
    segments = []
    current_segment = []

    with fitz.open(file_path) as doc:
        for page_number in range(len(doc)):
            # Check if this page is a break page
            if page_number + 1 in break_pages:
                # Add the current segment to segments (if it's not empty)
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []

            # Extract text from the current page
            page = doc.load_page(page_number)
            page_text = page.get_text()
            current_segment.append(page_text)

        # Add the last segment
        if current_segment:
            segments.append(' '.join(current_segment))

    return segments


# List of pages where each new document starts
break_pages = [7, 11, 15, 18, 24, 27, 31,  45, 49, 53, 57, 60, 65, 70, 76, 80, 85, 90, 98]  # Modify this list as needed

# Segment the PDF
pdf_segments = segment_pdf(pdf_path, break_pages)

# Associate each document (segment) with its region
document_data = [(segment, region) for segment, region in zip(pdf_segments, regions)]


# Displaying the first few characters of each segment to verify
segment_previews = {i+1: segment[:500] + '...' for i, segment in enumerate(pdf_segments)}

def preprocess_segments(segments):
    preprocessed_segments = []
    for segment in segments:
        # Tokenization
        from nltk.tokenize import word_tokenize

        from nltk.corpus import stopwords

        tokens = word_tokenize(segment)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        from nltk.stem import WordNetLemmatizer

        tokens = [token for token in tokens if token not in stop_words]

        # Removing punctuation
        tokens = [token for token in tokens if token.isalpha()]

        # Removing URLs
        tokens = [token for token in tokens if not re.match(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", token)]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Joining tokens back into text
        preprocessed_segment = ' '.join(tokens)
        preprocessed_segments.append(preprocessed_segment)
    
    return preprocessed_segments





# Preprocess the segments
from gensim import corpora, models
from fpdf import FPDF
from gensim.models import Word2Vec

#preprocessed_segments = preprocess_segments(pdf_segments)

# Preprocess the segments and maintain the regions
preprocessed_segments = preprocess_segments([doc for doc, _ in document_data])

# Keep track of regions after preprocessing
document_data_preprocessed = [(segment, region) for segment, (_, region) in zip(preprocessed_segments, document_data)]

'''
# Print out samples of each preprocessed segment text
for i, segment in enumerate(preprocessed_segments):
    print(f"Preprocessed Segment {i+1}:")
    print(segment[1000:1100])  # Print the first 100 characters of each segment
    print()

# Save preprocessed segments as a single PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

for i, segment in enumerate(preprocessed_segments):
    encoded_segment = segment.encode('latin-1', 'ignore').decode('latin-1')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, encoded_segment)
    pdf.add_page()

pdf.output("preprocessed_segments.pdf")
''' 
# Convert preprocessed segments to a list of tokenized documents
tokenized_documents = [segment.split() for segment in preprocessed_segments]
# Train the Word2Vec model
word2vec_model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)

# Get the word embeddings
word_embeddings = word2vec_model.wv
''' 
# Example usage
similar_words = word_embeddings.most_similar('example')
print(similar_words)
'''



## Updating orders of worth keywords through contextual similarity analysis:

# Orders of Worth Keywords
orders_of_worth_keywords = {
   
        "Inspirational": ["creativity", "inspiration", "art", "innovation", "imagination", "originality", "intuition", "passion", "spirit", "vision", "aesthetic", "muse", "genius", "artistic", "spontaneity", "expression", "idealism", "transcendence", "freedom", "inventiveness"],
        "Domestic": ["tradition", "family", "loyalty", "heritage", "trust", "kinship", "roots", "ancestry", "home", "lineage", "nurturing", "patriarchy", "matriarchy", "fidelity", "domesticity", "respect", "belonging", "elders", "customs", "genealogy", "family tree", "family history"],
        "Reputation": ["honor", "fame", "reputation", "prestige", "recognition", "status", "celebrity", "dignity", "glory", "esteem", "rank", "notoriety", "prominence", "distinction", "respectability", "nobility", "acclaim", "veneration", "admiration", "legacy"], #we see caste and honor and distinction here, interestingly not in domesitic
        "Civic": ["solidarity", "collective", "social", "public", "community", "welfare", "common good", "democracy", "civic", "participation", "equality", "justice", "citizenship", "public spirit", "social responsibility", "unity", "altruism", "cooperation", "commonwealth", "public interest", "social contract", "social cohesion", "social capital", "social justice"],
        "Market": ["competition", "market", "price", "value", "profit", "trade", "consumer", "business", "capitalism", "commerce", "transaction", "demand", "supply", "monetary", "investment", "bargaining", "enterprise", "economic", "marketing", "sales", "retail", "exchange", "commercial"],
        "Industrial": ["efficiency", "productivity", "technology", "expertise", "process", "skill", "method", "automation", "industry", "standardization", "precision", "proficiency", "workmanship", "specialization", "mechanism", "optimization", "craftsmanship", "technical", "output", "machinery", "manufacturing", "fabrication", "mechanization", "industrialization"],
        "Green": ["sustainability", "environment", "ecology", "conservation", "green", "renewable", "recycling", "biodiversity", "nature", "organic", "eco-friendly", "sustainable development", "climate change", "carbon footprint", "earth", "ecosystem", "natural resources", "pollution", "renewable energy", "conservationism"] #see iraq and violence here, connection btwn eco and interpersonal violence
    }

# Finding contextually similar words for each order
contextual_similarities = {}
for order, keywords in orders_of_worth_keywords.items():
    similar_words = []
    for keyword in keywords:
        if keyword in word2vec_model.wv:
            similar_words.extend([word for word, _ in word2vec_model.wv.most_similar(keyword, topn=3)])  # topn can be adjusted
    # Append the original keywords to the list of similar words
    similar_words.extend(keywords)
    contextual_similarities[order] = similar_words

# Update the lists with similar terms and remove terms not in word2vec list
for order, similar_words in contextual_similarities.items():
    orders_of_worth_keywords[order] = [word for word in similar_words if word in word2vec_model.wv]
    '''
        # Remove duplicated words from orders_of_worth_keywords
        for order, keywords in orders_of_worth_keywords.items():
            unique_keywords = []
            for keyword in keywords:
                if all(keyword not in orders_of_worth_keywords[other_order] for other_order in orders_of_worth_keywords if other_order != order):
                    unique_keywords.append(keyword)
    '''
# Convert each order list into a set and then back into a list (remove duplicates within each list but keep those between lists)
for order, keywords in orders_of_worth_keywords.items():
    unique_keywords = list(set(keywords))
    orders_of_worth_keywords[order] = unique_keywords


# Remove stopwords from the lists
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(["said", "many", "try", "must", "e", "ong", "new", "however", "formally", "aq", "want", "exist", "drop", "celebrate", "per", "also", "least"])  # Add additional stopwords as needed

orders_of_worth_keywords = {order: [word for word in keywords if word.lower() not in stopwords] for order, keywords in orders_of_worth_keywords.items()}

from collections import Counter

# Combine all documents into a single string and count word occurrences
all_words = ' '.join(preprocessed_segments).split()  # Preprocessed segments are already tokenized
word_counts = Counter(all_words)

# Check how often each keyword from the orders of worth appears
for order, keywords in orders_of_worth_keywords.items():
    keyword_counts = {kw: word_counts[kw] for kw in keywords if kw in word_counts}
    print(f'{order} keyword counts: {keyword_counts}')



# Print the updated lists
for order, keywords in orders_of_worth_keywords.items():
    print(f"Order: {order}")
    print(keywords)

    order_counts = {} 


# Initialize a list to store the normalized counts for each document
all_normalized_counts = []

# Count the number of times terms with each order appear in tokenized documents
for i, document in enumerate(tokenized_documents):
    print(f"Tokenized Document {i+1}:")
    document_counts = {}
    
    # Count occurrences of each keyword in the document for each order
    for order, keywords in orders_of_worth_keywords.items():
        order_count = sum(1 for term in document if term in keywords)
        document_counts[order] = order_count

    # Normalize the counts by adjusting for the length of the tokenized document
    normalized_counts = {}
    for order, count in document_counts.items():
        normalized_count = count / len(document)
        normalized_counts[order] = normalized_count
    
    # Store the normalized counts for this document
    all_normalized_counts.append(normalized_counts)


from collections import defaultdict

# Create a dictionary to store the aggregated normalized counts for each region
region_normalized_counts = defaultdict(lambda: defaultdict(list))

# Assuming `document_data_preprocessed` contains tuples of (preprocessed document, region)
for (document, region), normalized_counts in zip(document_data_preprocessed, all_normalized_counts):
    for order, count in normalized_counts.items():
        region_normalized_counts[region][order].append(count)

# Now, calculate the average normalized count for each order in each region
region_avg_normalized_counts = defaultdict(dict)
for region, order_counts in region_normalized_counts.items():
    for order, counts in order_counts.items():
        region_avg_normalized_counts[region][order] = np.mean(counts)  # Average over all documents in the region




# Assuming `document_data_preprocessed` contains tuples of (preprocessed document, region)
# and `all_normalized_counts` contains the normalized frequencies for each document

for i, ((document, region), normalized_counts) in enumerate(zip(document_data_preprocessed, all_normalized_counts)):
    print(f"Document {i+1}:")
    print(f"Region: {region}")
    print(f"Normalized Frequencies: {normalized_counts}")
    print("\n")



# Plot the average normalized counts for each region
import matplotlib.pyplot as plt

# Define a color map for each order
color_map = {
    "Green": "green",
    "Inspirational": "skyblue",
    "Domestic": "goldenrod",
    "Reputation": "darkorange",
    "Civic": "purple",
    "Market": "firebrick",
    "Industrial": "steelblue"
}

# Plot the average normalized counts for each region with the color scheme
for region, avg_counts in region_avg_normalized_counts.items():
    plt.figure(figsize=(10, 6))  # Set the figure size
    colors = [color_map[order] for order in avg_counts.keys()]  # Use the color map for each bar
    plt.bar(avg_counts.keys(), avg_counts.values(), color=colors)
    plt.xlabel('Order of Worth')
    plt.ylabel('Average Normalized Count')
    plt.title(f'Average Normalized Counts for {region}')
    plt.xticks(rotation=45)
    
    # Display the legend to explain colors
    handles = [plt.Rectangle((0,0),1,1, color=color_map[order]) for order in avg_counts.keys()]
    plt.legend(handles, avg_counts.keys(), title="Orders of Worth")
    
    plt.show()
    plt.close()




#Machine Learning Models:


from sklearn.feature_extraction.text import TfidfVectorizer

# Your preprocessed segments are the documents
documents = preprocessed_segments  # These are your preprocessed, tokenized text segments

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Adjust max_features as needed

# Fit and transform the preprocessed segments
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert to array (optional, depending on your needs)
tfidf_array = tfidf_matrix.toarray()

# Now, tfidf_array is ready for further analysis, clustering, etc.
print(tfidf_array)

from sklearn.cluster import AgglomerativeClustering, KMeans

# Specify the number of clusters (you can adjust K)
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(tfidf_matrix) 

# Get cluster labels
kmeans_labels = kmeans.labels_

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Standardize the data
scaler = StandardScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix.toarray())


# Optionally apply PCA to reduce dimensions
pca = PCA(n_components=3)
tfidf_matrix_scaled = pca.fit_transform(tfidf_matrix_scaled)

# Tune K-Means
OMP_NUM_THREADS=1




# Evaluate K-Means, Agglomerative Clustering, and GMM with different numbers of clusters for exploratory analysis
for n_clusters in range(2, 10):
    print(f"\nEvaluating {n_clusters} clusters")
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, max_iter=500, random_state=42)
    kmeans_labels = kmeans.fit_predict(tfidf_matrix_scaled)
    silhouette_kmeans = silhouette_score(tfidf_matrix_scaled, kmeans_labels)
    print(f"K-Means Silhouette score for {n_clusters} clusters: {silhouette_kmeans}")
    
    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_labels = agglomerative.fit_predict(tfidf_matrix_scaled)
    silhouette_agglomerative = silhouette_score(tfidf_matrix_scaled, agglomerative_labels)
    print(f"Agglomerative Clustering Silhouette score for {n_clusters} clusters: {silhouette_agglomerative}")
    
    # Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(tfidf_matrix_scaled)
    silhouette_gmm = silhouette_score(tfidf_matrix_scaled, gmm_labels)
    print(f"GMM Silhouette score for {n_clusters} clusters: {silhouette_gmm}")


# Assuming reduced_data is the 2D PCA-reduced data
# Apply K-Means with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42)
labels_2 = kmeans_2.fit_predict(tfidf_matrix_scaled)

# Apply K-Means with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42)
labels_3 = kmeans_3.fit_predict(tfidf_matrix_scaled)

# Plot the clusters for 2 clusters
plt.figure(figsize=(8, 6))
plt.scatter(tfidf_matrix_scaled[:, 0], tfidf_matrix_scaled[:, 1], c=labels_2, cmap='viridis', s=100, edgecolor='k')
plt.title('2 Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Plot the clusters for 3 clusters
plt.figure(figsize=(8, 6))
plt.scatter(tfidf_matrix_scaled[:, 0], tfidf_matrix_scaled[:, 1], c=labels_3, cmap='viridis', s=100, edgecolor='k')
plt.title('3 Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()








import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Apply PCA to reduce dimensions to 3
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())  # Use reduced PCA data

# Step 2: Apply K-Means with 3 clusters (or any number of clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_data)

# Step 3: Group and print document indices by their cluster
for cluster_id in set(kmeans_labels):  # Iterate over unique cluster labels
    print(f"\nDocuments in Cluster {cluster_id + 1}:\n")
    
    # Step 4: Loop through document indices and print those that belong to this cluster
    doc_indices = [i + 1 for i, label in enumerate(kmeans_labels) if label == cluster_id]
    # print(f"Document Indices: {doc_indices}")




from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 1: Prepare the features (X) and target variable (y)
X = reduced_data  # Use your TF-IDF matrix or PCA-reduced data
y = kmeans_labels           # Use cluster labels from K-Means as target

# Step 2: Split the data into training and testing sets (keeping this as-is for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

import numpy as np

# Assuming y_train and y_test contain the final cluster labels after splitting and training

# Combine train and test cluster labels to get the overall distribution
all_labels = np.concatenate((y_train, y_test))

# Check the distribution of documents across the clusters
unique, counts = np.unique(all_labels, return_counts=True)
cluster_distribution = dict(zip(unique, counts))

print("Cluster distribution across all documents:", cluster_distribution)




# Step 3: Define the models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Step 4: Use Stratified K-Fold Cross-Validation (3-Fold)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Train and evaluate the models with cross-validation
for model_name, model in models.items():
    # Perform 3-fold stratified cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
    print(f"{model_name} Stratified 3-Fold CV Accuracy: {np.mean(cv_scores):.4f}")

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}\n")

# Step 5: Hyperparameter tuning using GridSearchCV (with StratifiedKFold)
# Logistic Regression Tuning
log_reg_params = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l2']  # L2 regularization
}
log_reg_grid = GridSearchCV(LogisticRegression(random_state=42), log_reg_params, cv=skf)
log_reg_grid.fit(X_train, y_train)
print(f"Best parameters for Logistic Regression: {log_reg_grid.best_params_}")
print(f"Best CV accuracy for Logistic Regression: {log_reg_grid.best_score_:.4f}")

# Random Forest Tuning
rf_params = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Tree depth
    'min_samples_split': [2, 5, 10]   # Minimum samples to split a node
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=skf)
rf_grid.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {rf_grid.best_params_}")
print(f"Best CV accuracy for Random Forest: {rf_grid.best_score_:.4f}")

# SVM Tuning
svm_params = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # SVM kernel
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' kernel
}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=skf)
svm_grid.fit(X_train, y_train)
print(f"Best parameters for SVM: {svm_grid.best_params_}")
print(f"Best CV accuracy for SVM: {svm_grid.best_score_:.4f}")

# K-Nearest Neighbors Tuning
knn_params = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weighting of neighbors
    'p': [1, 2]  # Distance metric (1=Manhattan, 2=Euclidean)
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=skf)
knn_grid.fit(X_train, y_train)
print(f"Best parameters for K-Nearest Neighbors: {knn_grid.best_params_}")
print(f"Best CV accuracy for K-Nearest Neighbors: {knn_grid.best_score_:.4f}")





## Statistical Testing:
import numpy as np
from scipy.stats import chi2_contingency, f_oneway

# Prepare the contingency table, excluding the Middle Eastern region (document #19)
regions = [region for region in region_avg_normalized_counts.keys() if region != "Middle East"]  # Exclude "Middle East"
orders = list(orders_of_worth_keywords.keys())  # List of argument modes

# Build a contingency table (rows = regions, columns = argument modes)
contingency_table = []
for region in regions:
    region_row = []
    for order in orders:
        # Append the total count of the argument mode for this region
        region_row.append(sum(region_normalized_counts[region][order]))
    contingency_table.append(region_row)

# Convert to numpy array
contingency_table = np.array(contingency_table)

# Perform the chi-square test
chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)

# Output chi-square results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p_chi2}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies: \n{expected}")


# Perform ANOVA for each argument mode across regions (excluding the Middle Eastern region)
anova_results = {}

for order in orders:
    # Collect the normalized counts for this order across all regions, excluding "Middle East"
    region_counts = [region_normalized_counts[region][order] for region in regions if region != "Middle East"]
    
    # Perform one-way ANOVA for this argument mode across the remaining regions
    f_stat, p_value = f_oneway(*region_counts)  # The * unpacks the list into arguments
    
    # Store the results
    anova_results[order] = {
        "F-Statistic": f_stat,
        "P-Value": p_value
    }

# Output ANOVA results
for order, result in anova_results.items():
    print(f"Order: {order}")
    print(f"  F-Statistic: {result['F-Statistic']}")
    print(f"  P-Value: {result['P-Value']}")


from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

# Perform Tukey's HSD test for each argument mode
for order in orders:
    # Collect the normalized counts for this order across all regions, excluding "Middle East"
    values = []
    labels = []
    
    for region in regions:
        values.extend(region_normalized_counts[region][order])  # Add the argument mode counts
        labels.extend([region] * len(region_normalized_counts[region][order]))  # Add the corresponding region labels
    
    # Convert to numpy arrays
    values = np.array(values)
    labels = np.array(labels)
    
    # Perform Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
    
    # Print the results for this order
    print(f"Tukey HSD Results for {order}:")
    print(tukey)
    print("\n")


#Violin plots

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert `region_normalized_counts` into a long-form DataFrame
data = []
for region, counts in region_normalized_counts.items():
    for order, values in counts.items():
        for value in values:
            data.append({'Region': region, 'Order': order, 'Value': value})

# Create the DataFrame
df_long = pd.DataFrame(data)

# Plot the violin plot for each argument mode (order)
for order in orders:
    plt.figure(figsize=(10, 6))
    
    # Filter the DataFrame for the current order
    sns.violinplot(x='Region', y='Value', data=df_long[df_long['Order'] == order])
    
    plt.title(f'Violin Plot of {order} by Region')
    plt.show()

# no statisticially significant difference in the argumentation style across regions




##Linkages between argumentation modes


# Calculate the Spearman correlation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the list of dictionaries into a DataFrame for easy analysis
df = pd.DataFrame(all_normalized_counts)




# Calculate the Spearman correlation matrix
correlation_matrix = df.corr(method='spearman')

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=orders, yticklabels=orders)
plt.title("Spearman Correlation Between Argument Modes (Across Documents)")
plt.show()



# Group documents by region (assuming you have `document_data_preprocessed` that includes regions)
region_correlation_results = {}

for region in region_normalized_counts:
    if region != "Middle East":  # Exclude the Middle Eastern region
        # Create a DataFrame for each region
        df_region = pd.DataFrame(region_normalized_counts[region])
        
        # Calculate Spearman correlation for each region
        correlation_matrix_region = df_region.corr(method='spearman')
        
        # Store the result
        region_correlation_results[region] = correlation_matrix_region
        
        # Plot heatmap for each region
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix_region, annot=True, cmap="coolwarm", xticklabels=orders, yticklabels=orders)
        plt.title(f"Spearman Correlation Between Argument Modes ({region})")
        plt.show()


import pandas as pd

# Convert the list of normalized frequencies (all_normalized_counts) into a DataFrame
df = pd.DataFrame(all_normalized_counts)

# Add a 'Region' column based on the document regions from `document_data_preprocessed`
df['Region'] = [region for (_, region) in document_data_preprocessed]

# Display the DataFrame
print(df)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# Remove the 'Region' column before clustering (only argument modes are clustered)
df_clustering = df.drop(columns=['Region'])

# Perform hierarchical clustering
linkage_matrix = linkage(df_clustering.T, method='ward')  # Transpose (T) to cluster argument modes

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=df_clustering.columns)
plt.title('Hierarchical Clustering of Argument Modes')
plt.xlabel('Argument Modes')
plt.ylabel('Distance')
plt.show()


# Perform hierarchical clustering for each region separately
for region in df['Region'].unique():
    df_region = df[df['Region'] == region].drop(columns=['Region'])  # Filter for the region and drop the 'Region' column
    linkage_matrix = linkage(df_region.T, method='ward')  # Transpose to cluster argument modes

    # Plot the dendrogram for the region
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=df_region.columns)
    plt.title(f'Hierarchical Clustering of Argument Modes in {region}')
    plt.xlabel('Argument Modes')
    plt.ylabel('Distance')
    plt.show()


#Network Graph:
import networkx as nx

# Create a graph based on the Spearman correlation matrix

G = nx.Graph()

# Add edges for strong correlations (e.g., correlation > 0.5 or < -0.5)
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], 
                       weight=correlation_matrix.iloc[i, j])

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, edge_color='grey', width=2)
plt.title('Network of Argument Mode Correlations')
plt.show()



# Corex topic modeling:
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer




from sklearn.feature_extraction.text import CountVectorizer

# Your original dictionary of keyword lists
orders_of_worth_keywords = {
    'Inspirational': ['council', 'social', 'statement', 'conference', 'vision', 'artistic', 'expression', 'story', 'art', 'even', 'practice', 'treatment', 'woman', 'cultural', 'innovation', 'threshold', 'freedom', 'deemed', 'power', 'fraternity', 'dominion', 'appointment', 'pressure'],
    'Domestic': ['legislation', 'patriarchy', 'spirituality', 'associat', 'belonging', 'right', 'imsco', 'home', 'family', 'respect', 'slightly', 'education', 'stimulation', 'woman', 'indigenous', 'trust', 'heritage', 'cultural', 'state', 'resource', 'faith', 'word', 'article', 'tradition', 'perpetuated', 'global'],
    'Reputation': ['dignity', 'treating', 'legacy', 'social', 'infrastructure', 'engage', 'world', 'able', 'distinction', 'particular', 'increase', 'undertaken', 'blaming', 'honor', 'community', 'caste', 'tradition', 'extinction', 'peninsula', 'fair', 'recognition', 'delaware', 'pressing', 'reputation', 'ratio', 'status', 'american'],
    'Civic': ['economic', 'service', 'solidarity', 'destroys', 'social', 'technological', 'welfare', 'public', 'world', 'institute', 'international', 'collective', 'right', 'civil', 'participation', 'amapá', 'poverty', 'cooperation', 'health', 'local', 'action', 'equality', 'education', 'woman', 'member', 'indigenous', 'unity', 'community', 'people', 'justice', 'rural', 'particularly', 'followed', 'democracy', 'learns', 'girl', 'victim', 'society', 'violence', 'citizenship'],
    'Market': ['free', 'image', 'millennium', 'economic', 'owing', 'plagued', 'trade', 'investment', 'exchange', 'iraqi', 'social', 'massive', 'competition', 'national', 'effect', 'forget', 'international', 'powerless', 'discus', 'right', 'market', 'consumer', 'supply', 'paulo', 'business', 'demand', 'measure', 'appreciate', 'report', 'industry', 'action', 'profit', 'commercial', 'hope', 'conceive', 'artificial', 'woman', 'indigenous', 'including', 'cultural', 'enterprise', 'ect', 'community', 'computer', 'importance', 'people', 'resource', 'justice', 'replacement', 'institution', 'monetary', 'exemption', 'organization', 'value', 'pause', 'capitalism', 'price', 'entrance', 'change', 'statistical'],
    'Industrial': ['income', 'offered', 'social', 'transformation', 'principle', 'method', 'international', 'object', 'process', 'mechanism', 'based', 'agreement', 'industry', 'prioritizing', 'material', 'fabrication', 'amazonian', 'broad', 'people', 'resource', 'participant', 'upholding', 'technology', 'latest', 'august', 'skill', 'slave', 'girl', 'violence', 'caucus', 'expertise'],
    'Green': ['belgium', 'child', 'number', 'consumption', 'weakened', 'impeach', 'traditional', 'ecology', 'conservation', 'green', 'public', 'earth', 'recycling', 'enclosure', 'environment', 'group', 'agricultural', 'movement', 'right', 'hand', 'fight', 'iraq', 'renewable', 'role', 'biodiversity', 'work', 'reinforcement', 'subnational', 'period', 'indigenous', 'nature', 'pollution', 'sustainability', 'suffered', 'slain', 'source', 'ecosystem', 'violence', 'league', 'representation', 'sustainable']
}

# Step 1: Remove duplicates from each keyword list
orders_of_worth_keywords_no_duplicates = {
    order: list(dict.fromkeys(keywords)) for order, keywords in orders_of_worth_keywords.items()
}

# Step 2: Flatten the lists into a single list (if needed for CountVectorizer)
flattened_keywords = [' '.join(keywords) for keywords in orders_of_worth_keywords_no_duplicates.values()]

# Step 3: Pass the list to CountVectorizer
vectorizer = CountVectorizer()
doc_word = vectorizer.fit_transform(flattened_keywords)

# Print the feature names (vocabulary)
print(vectorizer.get_feature_names_out())


# Join the tokens back into a single string for each document
documents = [' '.join(doc) for doc in tokenized_documents]

# Convert the tokenized documents into a document-word matrix
doc_word = vectorizer.fit_transform(documents)
words = list(np.asarray(vectorizer.get_feature_names_out()))

#check sparsity
print(f'Document-Word Matrix Shape: {doc_word.shape}')
print(f'Number of Non-Zero Elements: {doc_word.nnz}')
sparsity = (doc_word.nnz / (doc_word.shape[0] * doc_word.shape[1])) * 100
print(f'Sparsity: {sparsity:.2f}%')


# Train the CorEx topic model with 7 topics (or adjust n_hidden based on your needs)
topic_model = ct.Corex(n_hidden=7, words=words, seed=1, anchor_strength=2)  # Set seed for reproducibility
topic_model.fit(doc_word, words=words, docs=documents)





# Print all topics from the CorEx topic model with word probabilities
topics = topic_model.get_topics()
for n, topic in enumerate(topics):
    if topic:  # Make sure topic is not empty
        topic_words = [word for word, *_ in topic]  # List of words, ignore extra elements
        word_probs = [prob for _, prob, *_ in topic]  # List of probabilities
        print(f'Topic {n+1}: {", ".join(topic_words)}')
        print(f'Word Probabilities: {word_probs}\n')
    else:
        print(f'Topic {n+1}: No anchor words found for this topic')

# Get the topic probabilities for each document
topic_probs = topic_model.p_y_given_x

# Print topic probabilities for each document
for i, probs in enumerate(topic_probs):
    print(f"Document {i+1} Topic Probabilities: {probs}")

# Convert topic probabilities to numpy array for easier handling
topic_probs = np.array(topic_probs)



# Create a DataFrame to store document-topic probabilities
doc_topic_df = pd.DataFrame(topic_probs, columns=[f'Topic {i+1}' for i in range(topic_probs.shape[1])])

# Add a column for document regions
doc_topic_df['Region'] =  [
    "Global",  # Document 1
    "Global",  # Document 2
    "Africa",  # Document 3
    "Latin America",  # Document 4
    "Asia",  # Document 5
    "Asia",  # Document 6
    "Global",  # Document 7
    "Latin America",  # Document 8
    "Global",  # Document 9
    "Global",  # Document 10
    "Global",  # Document 11
    "Global",  # Document 12
    "Global",  # Document 13
    "Asia",  # Document 14
    "Latin America",  # Document 15
    "Latin America",  # Document 16
    "Asia",  # Document 17
    "Global",  # Document 18
    "Middle East",  # Document 19
    "Africa"   # Document 20
]  # Assuming you have the regions list

# Display the DataFrame
print(doc_topic_df)

# Group by region and calculate the average topic probability per region
region_topic_avg = doc_topic_df.groupby('Region').mean()

for region in region_topic_avg.index:
    plt.figure(figsize=(10, 6))
    region_data = region_topic_avg.loc[region]
    region_data.plot(kind='bar', color='skyblue')
    plt.title(f'Topic Distribution in {region}')
    plt.xlabel('Topics')
    plt.ylabel('Average Topic Probability')
    plt.show()

# Plot heatmap of topic probabilities by document
plt.figure(figsize=(10, 8))
sns.heatmap(topic_probs, annot=True, cmap="Blues", xticklabels=[f'Topic {i+1}' for i in range(topic_probs.shape[1])])
plt.xlabel('Topics')
plt.ylabel('Documents')
plt.title('Document-Topic Probabilities')
plt.show()















