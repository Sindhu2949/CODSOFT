# ==================================================
# HANDWRITTEN TEXT GENERATION USING LSTM
# Internship Project - Improved Version
# ==================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

import tensorflow as tf
import numpy as np

# ===============================
# 1️⃣ Load Dataset
# ===============================

print("Loading dataset...")

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# 🔥 Use only first 1M characters for faster CPU training
text = text[:1000000]

print("Dataset length used:", len(text))

# ===============================
# 2️⃣ Create Vocabulary
# ===============================

vocab = sorted(set(text))
print("Vocabulary size:", len(vocab))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# ===============================
# 3️⃣ Create Sequences
# ===============================

seq_length = 100

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 5000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print("Training samples ready!")

# ===============================
# 4️⃣ Build Improved Model
# ===============================

vocab_size = len(vocab)
embedding_dim = 192
rnn_units = 192

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# ===============================
# 5️⃣ Train Model
# ===============================

EPOCHS = 8

history = model.fit(dataset, epochs=EPOCHS)

# ===============================
# 6️⃣ Text Generation Function
# ===============================

def generate_text(model, start_string, num_generate=400, temperature=0.5):

    input_eval = [char2idx[s] for s in start_string.lower() if s in char2idx]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    for i in range(num_generate):

        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# ===============================
# 7️⃣ Generate Text
# ===============================

print("\nGenerating Text...\n")

generated_text = generate_text(
    model,
    start_string="The handwritten text says ",
    num_generate=400,
    temperature=0.5
)

print("Generated Text:\n")
print(generated_text)
