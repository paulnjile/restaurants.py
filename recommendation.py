import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read restaurant data from the CSV file
restaurants = []
with open('/content/Restaurants.csv', 'r', encoding='latin-1') as file:
    reader = csv.DictReader(file)
    for row in reader:
        restaurants.append(row)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in restaurants:
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

# Now, suppose a user likes "Restaurant 1". We can recommend another restaurant based on cosine similarity.
liked_restaurant = "Zanzibar Manta Resort"

try:
    liked_restaurant_index = next(index for index, restaurant in enumerate(restaurants) if restaurant['TITLE'] == liked_restaurant)
    # Find the most similar restaurant
    similar_restaurant_index = similarity_matrix[liked_restaurant_index].argsort()[::-1][1]  # Exclude the liked restaurant itself
    recommended_restaurant = restaurants[similar_restaurant_index]
    print("Because you liked " + liked_restaurant + ", we recommend: " + recommended_restaurant['TITLE'])
except StopIteration:
    print("Liked restaurant not found in the list.")
