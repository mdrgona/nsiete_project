from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate

class JokeRecommender(keras.Model):
    
    def __init__(self, user_input, joke_input, units1, units2):
        super(JokeRecommender, self).__init__(name='joke_recommender')
        
        # Input layers
        user_input = Input(shape=(1,), dtype='int32')
        joke_input = Input(shape=(1,), dtype='int32')
                
        # Embedding layers
        user_embedding = Embedding(input_dim=len(user_input), output_dim=int(units1 / 2), input_length=1)
        joke_embedding = Embedding(input_dim=len(joke_input), output_dim=int(units1 / 2), input_length=1)
        
        # Concatenate user and joke embeddings
        user_flatten = Flatten()(emb_user(user_input))
        joke_flatten = Flatten()(emb_item(joke_input))
        final_vector = concatenate([user_flatten, joke_flatten])
        
        print(final_vector)
        
        self.layer_1 = Dense(units=units1, activation='relu')
        self.layer_2 = Dense(units=units2, activation='relu')
        

    def call(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x