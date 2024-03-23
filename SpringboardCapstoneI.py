import PyPDF2
from PyPDF2 import PdfReader
import re

import fitz

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Example usage
pdf_path = "NGOTextFile.pdf"
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
break_pages = [7, 11, 15, 18, 24, 27, 31, 36, 40, 45, 49, 53, 57, 60, 65, 70, 76, 80, 85, 90, 98]  # Modify this list as needed

# Segment the PDF
pdf_segments = segment_pdf(pdf_path, break_pages)

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

preprocessed_segments = preprocess_segments(pdf_segments)

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

# Convert preprocessed segments to a list of tokenized documents
tokenized_documents = [segment.split() for segment in preprocessed_segments]
# Train the Word2Vec model
word2vec_model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)

# Get the word embeddings
word_embeddings = word2vec_model.wv

# Example usage
similar_words = word_embeddings.most_similar('example')
print(similar_words)

word_embeddings = word2vec_model.wv
# Orders of Worth Keywords
orders_of_worth_keywords = {
   
        "Inspirational": ["creativity", "inspiration", "art", "innovation", "imagination", "originality", "intuition", "passion", "spirit", "vision", "aesthetic", "muse", "genius", "artistic", "spontaneity", "expression", "idealism", "transcendence", "freedom", "inventiveness"],
        "Domestic": ["tradition", "family", "loyalty", "heritage", "trust", "kinship", "roots", "ancestry", "home", "lineage", "nurturing", "patriarchy", "matriarchy", "fidelity", "domesticity", "respect", "belonging", "elders", "customs", "genealogy", "family tree", "family history"],
        "Reputation": ["honor", "fame", "reputation", "prestige", "recognition", "status", "celebrity", "dignity", "glory", "esteem", "rank", "notoriety", "prominence", "distinction", "respectability", "nobility", "acclaim", "veneration", "admiration", "legacy"],
        "Civic": ["solidarity", "collective", "social", "public", "community", "welfare", "common good", "democracy", "civic", "participation", "equality", "justice", "citizenship", "public spirit", "social responsibility", "unity", "altruism", "cooperation", "commonwealth", "public interest", "social contract", "social cohesion", "social capital", "social justice"],
        "Market": ["competition", "market", "price", "value", "profit", "trade", "consumer", "business", "capitalism", "commerce", "transaction", "demand", "supply", "monetary", "investment", "bargaining", "enterprise", "economic", "marketing", "sales", "retail", "exchange", "commercial"],
        "Industrial": ["efficiency", "productivity", "technology", "expertise", "process", "skill", "method", "automation", "industry", "standardization", "precision", "proficiency", "workmanship", "specialization", "mechanism", "optimization", "craftsmanship", "technical", "output", "machinery", "manufacturing", "fabrication", "mechanization", "industrialization"],
        "Green": ["sustainability", "environment", "ecology", "conservation", "green", "renewable", "recycling", "biodiversity", "nature", "organic", "eco-friendly", "sustainable development", "climate change", "carbon footprint", "earth", "ecosystem", "natural resources", "pollution", "renewable energy", "conservationism"]
    }


# Finding contextually similar words for each order
contextual_similarities = {}
for order, keywords in orders_of_worth_keywords.items():
    similar_words = []
    for keyword in keywords:
        if keyword in word2vec_model.wv:
            similar_words.extend([word for word, _ in word2vec_model.wv.most_similar(keyword, topn=5)])  # topn can be adjusted
    contextual_similarities[order] = similar_words

# Update the lists with similar terms and remove terms not in word2vec list
for order, similar_words in contextual_similarities.items():
    orders_of_worth_keywords[order] = [word for word in similar_words if word in word2vec_model.wv]


# Print the updated lists
for order, keywords in orders_of_worth_keywords.items():
    print(f"Order: {order}")
    print(keywords)

    order_counts = {}

    # Count the number of times terms with each order appear in tokenized documents
    for i, document in enumerate(tokenized_documents):
        print(f"Tokenized Document {i+1}:")
        document_counts = {}
        for order, keywords in orders_of_worth_keywords.items():
            order_count = sum(1 for term in document if term in keywords)
            document_counts[order] = order_count

        # Normalize the counts by adjusting for the length of the tokenized document
        normalized_counts = {}
        for order, count in document_counts.items():
            normalized_count = count / len(document)
            normalized_counts[order] = normalized_count

        # Print the normalized counts for the document
        import matplotlib.pyplot as plt

   
        # Plot the normalized counts for each document
        for i, document in enumerate(tokenized_documents):
            plt.figure()
            plt.bar(orders_of_worth_keywords.keys(), [normalized_counts[order] for order in orders_of_worth_keywords.keys()])
            plt.xlabel('Order of Worth')
            plt.ylabel('Normalized Count')
            plt.title(f'Normalized Counts for Document {i+1}')
            plt.show()

'''
print(similar_words)
# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(tokenized_documents)

# Convert the corpus into Bag of Words format
bow_corpus = [dictionary.doc2bow(text) for text in tokenized_documents]

# Convert the tokenized documents to TF-IDF representation
tfidf_model = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf_model[bow_corpus]

# Train the LDA model
num_topics = 5  # Specify the number of topics you want to extract
lda_model = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=10)

# Get the topics and their corresponding keywords
topics = lda_model.print_topics(num_topics=num_topics)

# Print the topics

for topic in topics:
    print(topic)

    # Classify each segment into a topic
segment_topics = []
for segment in tokenized_documents:
    # Convert the segment to a TF-IDF representation
    segment_tfidf = tfidf_model[dictionary.doc2bow(segment)]
    # Get the topic distribution for the segment
    segment_topic = lda_model.get_document_topics(segment_tfidf)
    # Sort the topics by their probability in descending order
    segment_topic = sorted(segment_topic, key=lambda x: x[1], reverse=True)
    # Get the most probable topic for the segment
    most_probable_topic = segment_topic[0][0]
    segment_topics.append(most_probable_topic)

# Print the segment and its corresponding topic
for i, segment in enumerate(pdf_segments):
    print(f"Segment {i+1}:")
  #  print(segment)
    print(f"Topic: {segment_topics[i]}")
    print()
'''
