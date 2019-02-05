import numpy as np

########################### 1. READ IN INPUT DATA #############################
with open('data/small_vocab_en', 'r') as f:
    eng_sentences = f.read().split('\n')
    
with open('data/small_vocab_fr', 'r') as f:
    fre_sentences = f.read().split('\n')
    
print("Number of English-frenish Pairs: ", len(eng_sentences))
print("Sample Pair: \n")
print(eng_sentences[0])
print(fre_sentences[0])
print()


######################### 2. PRE-PROCESS INPUT TEXT ###########################
from keras.preprocessing.text import Tokenizer

def tokenize(sentences, encode_start_end = False):
    
    if encode_start_end:
        sentences = ["startofsentence " + s + "endofsentence" for s in sentences]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)
    
    return tokenized_sentences, tokenizer


from keras.preprocessing.sequence import pad_sequences

def pad(sentences, length = None):
    
    if length is None:
        length = max([len(s) for s in sentences])
        
    padded_sentences = pad_sequences(sentences, 
                                     maxlen = length,
                                     padding = 'post',
                                     truncating = 'post')

    return padded_sentences


eng_tokenized, eng_tokenizer = tokenize(eng_sentences)
fre_tokenized, fre_tokenizer = tokenize(fre_sentences,
                                        encode_start_end = True)

eng_encoded = pad(eng_tokenized)
fre_encoded = pad(fre_tokenized)

eng_vocab_size = len(eng_tokenizer.word_index)
fre_vocab_size = len(fre_tokenizer.word_index)

print("English vocabulary size: ", eng_vocab_size)
print("frenish vocabulary size: ", fre_vocab_size)
print()
  
eng_seq_len = len(eng_encoded[0])
fre_seq_len = len(fre_encoded[0])
        
print("Length of longest English sentence: ", eng_seq_len)
print("Length of longest frenish sentence: ", fre_seq_len)
print()
        

######################### 3. BUILD *TRAINING* MODEL ###########################
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import frerse_categorical_crossentropy

encoder_input = Input(shape = (None, ),
                      name = "Encoder_Input")

embedding_dim = 200
embedded_input = Embedding(input_dim = eng_vocab_size,
                           output_dim = embedding_dim,
                           name = "Embedding_Layer")(encoder_input)

encoder_lstm = LSTM(units = 256,
                    activation = "relu",
                    return_sequences = False,
                    return_state = True,
                    name = "Encoder_LSTM")

_, last_h_encoder, last_c_encoder = encoder_lstm(embedded_input)

decoder_input = Input(shape = (None, 1),
                      name = "Deocder_Input")

decoder_lstm = LSTM(units = 256,
                    activation = "relu",
                    return_sequences = True,
                    return_state = True,
                    name = "Decoder_LSTM")

all_h_decoder, _, _ = decoder_lstm(decoder_input,
                                   initial_state = [last_h_encoder, last_c_encoder])

final_dense = Dense(fre_vocab_size,
                    activation = 'softmax',
                    name = "Final_Dense_Layer")

logits = final_dense(all_h_decoder)

seq2seq_model = Model(input = [encoder_input, decoder_input],
                      output = logits)

seq2seq_model.compile(loss = frerse_categorical_crossentropy,
                      optimizer = Adam(lr = 0.002),
                      metrics = ['accuracy'])


############################ 4. TRAIN THE MODEL ###############################
# Decoder: input - all but last word, target - all but "starofsentence" token
decoder_fre_input = fre_encoded.reshape((-1, fre_seq_len, 1))[:, :-1, :]
decoder_fre_target = fre_encoded.reshape((-1, fre_seq_len, 1))[:, 1:, :]

seq2seq_model.fit([eng_encoded, decoder_fre_input],
                  decoder_fre_target,
                  epochs = 16,
                  batch_size = 1024)
        
        
######################### 5. BUILD *INFERENCE* MODEL ##########################
inf_encoder_model = Model(input = encoder_input, 
                          output = [last_h_encoder, last_c_encoder])

decoder_initial_states = [Input(shape = (256,)), 
                          Input(shape = (256,))]

all_h_decoder, last_h_decoder, last_c_decoder = decoder_lstm(decoder_input,
                                                             initial_state = decoder_initial_states)

logits = final_dense(all_h_decoder)

inf_decoder_model = Model(input = decoder_input + decoder_initial_states,
                          output = logits + [last_h_decoder, last_c_decoder])


############################### 6. TRANSLATE!! ################################
# word id -> word dict for frenish:
fre_id2word = {idx:word for word, idx in fre_tokenizer.word_index.items()}

def translate(eng_sentence):
    eng_sentence = eng_sentence.reshape((1, eng_seq_len))  # give batch size of 1
    initial_states = inf_encoder_model.predict(eng_sentence)
    # Initialize decoder input as a length 1 sentence containing "startofsentence",
    # --> feeding the start token as the first predicted word
    prev_word = np.zeros((1,1,1))
    prev_word[0, 0, 0] = fre_tokenizer.word_index["startofsentence"]

    stop_condition = False
    translation = []
    while not stop_condition:
        # 1. predict the next word using decoder model
        logits, last_h, last_c = inf_decoder_model.predict([prev_word] + initial_states)
        
        # 2. Update prev_word with the predicted word
        predicted_id = np.argmax(logits[0, 0, :])
        predicted_word = fre_id2word[predicted_id]
        translation.append(predicted_word)

        # 3. Enable End Condition: (1) if predicted word is "endofsentence" OR
        #                          (2) if translated sentence reached maximum sentence length
        if (predicted_word == 'endofsentence' or len(translation) > fre_seq_len):
            stop_condition = True

        # 4. Update prev_word with the predicted word
        prev_word[0, 0, 0] = predicted_id

        # 5. Update initial_states with the previously predicted word's encoder output
        initial_states = [last_h, last_c]

    return " ".join(translation).replace('endofsentence', '')
        

for i in [1, 100, 10000, 50000]:
    frenish_translation = translate(eng_encoded[i])
    print(eng_sentences[i])
    print(frenish_translation)
    print(fre_sentences[i])
      