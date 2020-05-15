import tensorflow as tf
import numpy as np
import os

# load and lightly pre-process data
text = " ".join(open("trump_tweets_all.txt").readlines())
text = " ".join(text.split())
text = text.encode("ascii", errors="ignore").decode()
print(text[:100])

# calculate the vocabulary (number of unique characters in text)
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# mapping from unique characters to indices
char_to_index = {u: i for i, u in enumerate(vocab)}
index_to_char = np.array(vocab)

# numerical representation of text
text_as_int = np.array([char_to_index[c] for c in text])

# define the sequence length, which will determine how many example pairs per epoch
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)


def split_input_target(chunk):
    """Split a chunk of length n+1 into a tuple containing the the input (first n chars)
        and the target (last n chars) """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# create the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 64
rnn_units = 512


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Define the model: character embedding -> GRU -> fully connected """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


train = False
if train:
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    model.summary()

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    # attach optimizer and loss
    model.compile(optimizer='adam', loss=loss)

    # Set up directory for saving checkpoints of the model
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 30
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


def generate_text(model, start_string):
    """Generate text, given a trained model and a starting string"""
    num_generate = 280
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])

    return start_string + ''.join(text_generated)


# set generate to True to generate text
generate = True
if generate:
    # load the model back from a checkpoint
    checkpoint_dir = './training_checkpoints'
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    print(generate_text(model, start_string="Hillary"))