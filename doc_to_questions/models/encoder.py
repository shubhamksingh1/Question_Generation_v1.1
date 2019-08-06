import tensorflow as tf

def encoder_model(embedding, EMBEDDING_DIMENS, hparams):
    """ takes sentences as an input and passes last state of its layer to initial state of decoder """

    encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoder_inputs")
    encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")

    encoder_emb_inp = tf.nn.embedding_lookup(embedding, encoder_inputs)

    encoder_forward_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), hparams.keep_prob) for _ in range(int(hparams.num_layers / 2))])
    encoder_backward_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), hparams.keep_prob) for _ in range(int(hparams.num_layers / 2))])

    bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_forward_cell, encoder_backward_cell,
                                                                   encoder_emb_inp, sequence_length=encoder_lengths,
                                                                   dtype=tf.float32)
    encoder_output = tf.concat(bi_outputs, -1)

    encoder_state = []
    for layer_id in range(1):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)

    return encoder_inputs, encoder_lengths, encoder_state, encoder_output
