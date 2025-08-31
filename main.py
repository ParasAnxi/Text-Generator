import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file(
    'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]
characters = sorted(set(text))

char_to_idx = dict((c, i) for i, c in enumerate(characters))
idx_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_idx[character]] = 1
    y[i, char_to_idx[next_characters[i]]] = 1

# train model
# model = Sequential()
# model.add(LSTM(128, input_shape = (SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01))
# model.fit(x, y,batch_size=256, epochs=4)
# model.save('textgenerator.keras')

model = tf.keras.models.load_model('textgenerator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


def generate_text(length, temperature=1.0):
    start_idx = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated_text = ""
    sentence = text[start_idx: start_idx + SEQ_LENGTH]
    generated_text += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_idx[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_idx = sample(predictions, temperature)
        next_character = idx_to_char[next_idx]
        generated_text += next_character
        sentence = sentence[1:] + next_character
    return generated_text


print("-----------------0.2-----------------")
print(generate_text(300, 0.2))
print("-----------------0.6-----------------")
print(generate_text(300, 0.6))
print("-----------------0.8-----------------")
print(generate_text(300, 0.8))
print("-----------------1-----------------")
print(generate_text(300, 1.0))