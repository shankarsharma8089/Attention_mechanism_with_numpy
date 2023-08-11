import numpy as np

# Define the sentence
sentence = "i miss the old kanye"

# Define the vocabulary
vocabulary = set(sentence.split())

# Define the word embeddings
word_embeddings = {}
for word in vocabulary:
    word_embeddings[word] = np.random.rand(3)  # Generate random embeddings

# Encode the sentence as a sequence of word embeddings
encoded_sentence = [word_embeddings[word] for word in sentence.split()]

# Assign the encoded sentence to variables
i = encoded_sentence[0]
miss = encoded_sentence[1]
the = encoded_sentence[2]
old = encoded_sentence[3]
kanye = encoded_sentence[4]

# Print the variables
print('Creating word embedding')
print(f"i: {i}")
print(f"miss: {miss}")
print(f"the: {the}")
print(f"old: {old}")
print(f"kanye: {kanye}")

# word embeddings.
words = np.array([i , miss , the , old , kanye])

#Next, we generate the weight matrices that will be multiplied with the word embeddings to obtain the queries, keys, and values. For this example, we randomly generate these weight matrices, but in real scenarios, they would be learned during training.
np.random.seed(42) 
W_Query = np.random.randint(3, size=(3, 3))
W_Key = np.random.randint(3, size=(3, 3))
W_Value = np.random.randint(3, size=(3, 3))

# Generating the queries, keys, and values.
Q = np.dot(words, W_Query)
K = np.dot(words, W_Key)
V = np.dot(words, W_Value)

# Scoring vector query.
scores = np.dot(Q, K.T)

# Computing the weights by applying a softmax operation.
weights = softmax(scores / np.sqrt(K.shape[1]), axis=1)

# Computing the attention by calculating the weighted sum of the value vectors.
attention = np.dot(weights, V)
print("Attention")
print(attention)

#output
Creating word embedding
i: [0.46676289 0.85994041 0.68030754]
miss: [0.52477466 0.39986097 0.04666566]
the: [0.45049925 0.01326496 0.94220176]
old: [0.97375552 0.23277134 0.09060643]
kanye: [0.61838601 0.38246199 0.98323089]
Attention
[[0.81676354 1.14873475 0.33197121]
 [0.71861207 1.07919157 0.3605795 ]
 [0.74203343 1.13656778 0.39453435]
 [0.75607715 1.11278575 0.3567086 ]
 [0.79709991 1.15043165 0.35333174]]
