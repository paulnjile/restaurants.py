
import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read movie data from the CSV file
movies = []
with open('/MOVIES2.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        movies.append(row)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in movies:
    description = row['DESCRIPTION']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Movie 1". We can recommend another movie based on cosine similarity.
liked_movie = "The Shawshank Redemption"
liked_movie_index = next(index for index, movie in enumerate(movies) if movie['TITLE'] == liked_movie)

# Find the most similar movie
similar_movie_index = similarity_matrix[liked_movie_index].argsort()[::-1][1]  # Exclude the liked movie itself
recommended_movie = movies[similar_movie_index]

print("Because you liked " + liked_movie + ", we recommend: " + recommended_movie['TITLE'])
