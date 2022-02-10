from tensorflow.keras.layers import Embedding
import numpy as np

def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    - word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    - word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    - embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_size = len(word_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)

    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_size,emb_dim))
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_size,emb_dim,trainable=False)
    
    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def sentences_to_indices(X,word_to_index,max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to 'Embedding()' (described in Figure 4). 
    
    Arguments:
    - X -- array of sentences (strings), of shape (m, 1)
    - word_to_index -- a dictionary containing the each word mapped to its index
    - max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    - X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    n_samples = X.shape[0] # number of training examples
    
    X_indices = np.zeros((n_samples,max_len))
    
    for i in range(n_samples):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split it into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # if w exists in the word_to_index dictionary
            if w in word_to_index:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i,j] = word_to_index[w]
                # Increment j to j + 1
                j += 1
    
    return X_indices