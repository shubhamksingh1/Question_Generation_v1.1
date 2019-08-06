import tensorflow as tf
from tensorflow.python.layers.core import Dense

def decoder_model(embedding, EMBEDDING_DIMENS, encoder_output, encoder_lengths, encoder_state, hparams):

    

    decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
    decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")

    decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_inputs)

    projection_layer = Dense(embedding.shape[0], use_bias=False)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(EMBEDDING_DIMENS, encoder_output,
                                                            memory_sequence_length=encoder_lengths)

    decoder_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), hparams.keep_prob) for _ in
         range(hparams.num_layers)])
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                       attention_layer_size=EMBEDDING_DIMENS)
    decoder_initial_state = decoder_cell.zero_state(hparams.batch_size, dtype=tf.float32).clone(cell_state=encoder_state)

    

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([1], 1), 2)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                              output_layer=projection_layer)
    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    decoder_outputs = decoder_outputs.rnn_output

    return decoder_outputs