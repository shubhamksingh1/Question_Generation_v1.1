import pandas as pd
import numpy as np
import nltk
import itertools
from models.encoder import encoder_model
from models.decoder import decoder_model
import tensorflow as tf
import os
import logging
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directo

from collections import Counter
from tensorflow.python.layers.core import Dense
from IPython.display import clear_output
import itertools

logger = logging.getLogger()
logger.setLevel(logging.ERROR)




def data_preprocessing(file_name):

	SENTENCES = []

	LOAD_DATA = pd.read_csv(file_name, engine="python", sep="\t", header=None)

	# get answers/paragraphs from data
	PARAGRAPHS = LOAD_DATA.iloc[:,1:2].values

	for paras in PARAGRAPHS:
	
		SEG_PARA = nltk.sent_tokenize(paras[0])
		
		SENTENCES.extend(SEG_PARA)
	print("Preprocessing**********",SENTENCES)

	return SENTENCES


def qg_engine(file_name):

	SENTENCES = data_preprocessing("output/"+file_name+"_qa.csv")
	print("*******************RETURNED from preprocessing\n",SENTENCES)

	tf.reset_default_graph()

	"""# **CREATE EMBEDDINGS**"""

	chars = []
	embeddings = []

	data = pd.read_csv("/Users/Anurag/Downloads/gen-questions-master/Data/char_embedding.csv", engine="python", header = None, sep="\t")

	# get characters from dataset
	char_list = data.iloc[:, 0:1].values

	for i in range(len(char_list)):
	    chars.append(char_list[i][0])

	# get embeddings of respective characters
	char_embeddings = data.iloc[:, 1:-1].values

	for i in range(len(char_embeddings)):
	    embeddings.append(char_embeddings[i])

	"""##**CREATE EMBEDDINGS ARRAY**"""

	embeddings_array = np.ndarray((73, 512), dtype=np.float32)

	char_to_idx = {}
	idx_to_char = []

	def add_char(char):
	    idx = len(idx_to_char)
	    char_to_idx[char] = idx
	    idx_to_char.append(char)
	    return idx

	UNKNOWN_CHAR = "<UNK>"
	START_CHAR = "<START>"
	END_CHAR = "<END>"

	UNKNOWN_TOKEN = add_char(UNKNOWN_CHAR)
	START_TOKEN = add_char(START_CHAR)
	END_TOKEN = add_char(END_CHAR)

	def look_up_char(char):
	    return char_to_idx.get(char, UNKNOWN_TOKEN)

	def look_up_token(token):
	    return idx_to_char[token]

	embeddings_array[UNKNOWN_TOKEN] = np.zeros(512)
	embeddings_array[START_TOKEN] = -np.ones(512)
	embeddings_array[END_TOKEN] = np.ones(512)

	for i, char in enumerate(chars):
	    idx = add_char(char)
	    embeddings_array[idx] = embeddings[i]

	"""#**HYPERPARAMETERS**"""

	batch_size = len(SENTENCES)
	keep_prob = 0.75
	num_layers = 2
	EPOCHS = 500
	# EPOCHS = 5
	learning_rate = 0.0005
	max_gradient_norm = 5

	"""#**DATA PREPROCESSING AND PREPARING BATCH**"""

	def tokenize(string):
	    return list(string)

	def prepare_batch(batch):
	    id_to_indices = {}
	    sentence_id = []
	    sentence_text = []
	    sentence_chars = []
	    question_text = []
	    question_input_chars = []
	    question_output_chars = []  
	    for i, entry in enumerate(batch):
	        id_to_indices.setdefault(entry["sentence_id"], []).append(i)
	        sentence_id.append(entry["sentence_id"])
	        sentence_text.append(entry["sentence_text"])
	        sentence_chars.append(entry["sentence_chars"])
	        question_text.append(entry["question_text"])
	        question_chars = entry["question_chars"]
	        question_input_chars.append([START_CHAR] + question_chars)
	        question_output_chars.append(question_chars + [END_CHAR])
	    batch_size_prepare = len(batch)
	    max_sentence_length = max((len(sentence) for sentence in sentence_chars), default=0)
	    max_question_length = max((len(question) for question in question_input_chars), default=0)
	  
	    sentence_tokens = np.zeros((batch_size_prepare, max_sentence_length), dtype = np.int32)
	    sentence_lengths = np.zeros(batch_size_prepare, dtype = np.int32)
	    question_input_tokens = np.zeros((batch_size_prepare, max_question_length), dtype = np.int32)
	    question_output_tokens = np.zeros((batch_size_prepare, max_question_length), dtype = np.int32)
	    question_lengths = np.zeros(batch_size_prepare, dtype = np.int32)
	  
	    for i in range(batch_size_prepare):
	        for j, char in enumerate(sentence_chars[i]):
	            sentence_tokens[i, j] = look_up_char(char)
	        sentence_lengths[i] = len(sentence_chars[i])
	        for j, char in enumerate(question_input_chars[i]):
	            question_input_tokens[i, j] = look_up_char(char)
	        for j, char in enumerate(question_output_chars[i]):
	            question_output_tokens[i, j] = look_up_char(char)
	        question_lengths[i] = len(question_input_chars[i])
	    
	    return {
	      "size": batch_size_prepare,
	      "sentence_id": sentence_id,
	      "sentence_tokens": sentence_tokens,
	      "sentence_lengths": sentence_lengths,
	      "sentence_text": sentence_text,
	      "question_input_tokens": question_input_tokens,
	      "question_output_tokens": question_output_tokens,
	      "question_lengths": question_lengths,
	      "question_text": question_text
	    }

	def process_batch(sentence_question):
	    batch = []
	    for text in sentence_question.values():
	        if len(batch) + len(text) > batch_size:
	            yield prepare_batch(batch)
	            batch = []
	        batch.extend(text)
	    if batch:
	        yield prepare_batch(batch)

	def read_data(sentences, questions):
	    sentence_question = {}
	    for i, (sentence, question) in enumerate(zip(sentences, questions)):
	        sentence_id = i
	        existing = sentence_question.setdefault(sentence_id, [])
	        sentence_text = sentence
	        sentence_chars = tokenize(sentence)
	        question_text = question
	        question_chars = tokenize(question)
	        existing.append({
	            "sentence_id": sentence_id,
	            "sentence_text": sentence_text,
	            "sentence_chars": sentence_chars,
	            "question_text": question_text,
	            "question_chars": question_chars
	        })
	    return sentence_question

	def training_data(sentences, questions):
	    return process_batch(read_data(sentences, questions))

	"""#**INITIALIZE EMBEDDINGS**"""

	embedding = tf.get_variable("embedding", initializer=embeddings_array)
	EMBEDDING_DIMENS = embeddings_array.shape[1]

	"""#**ENCODER MODEL**"""

	encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoder_inputs")
	encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")

	encoder_emb_inp = tf.nn.embedding_lookup(embedding, encoder_inputs)

	encoder_forward_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), keep_prob) for _ in range(int(num_layers / 2))])
	encoder_backward_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), keep_prob) for _ in range(int(num_layers / 2))])

	bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_forward_cell, encoder_backward_cell, encoder_emb_inp, sequence_length=encoder_lengths, dtype=tf.float32)
	encoder_output = tf.concat(bi_outputs, -1)

	encoder_state = []
	for layer_id in range(int(num_layers / 2)):
	    encoder_state.append(bi_encoder_state[0][layer_id])
	    encoder_state.append(bi_encoder_state[1][layer_id])
	encoder_state = tuple(encoder_state)

	"""#**Training Decoder Model**"""

	decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
	decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")

	decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_inputs)

	projection_layer = Dense(embedding.shape[0], use_bias=False)

	attention_mechanism = tf.contrib.seq2seq.LuongAttention(EMBEDDING_DIMENS, encoder_output, memory_sequence_length=encoder_lengths)

	decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EMBEDDING_DIMENS), keep_prob) for _ in range(num_layers)])
	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=EMBEDDING_DIMENS)
	decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=encoder_state)

	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)

	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=projection_layer)


	decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
	decoder_outputs = decoder_outputs.rnn_output

	decoder_labels = tf.placeholder(tf.int32, shape=[None, None], name="decoder_labels")
	question_mask = tf.sequence_mask(decoder_lengths, dtype=tf.float32)

	"""#**Loss, Gradient Clipping and Optimizer**"""

	loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs, targets=decoder_labels, weights=question_mask, name="loss")

	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

	learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
	optimizer = optimizer.apply_gradients(zip(clipped_gradients, params))

	"""#**Session and Saver**"""

	saver = tf.train.Saver()
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())
	saver.restore(session, "/Users/Anurag/Downloads/gen-questions-master/doc_to_questions/pretrained_model/model_const_lr-500")

	"""#**Testing Decoder Model**"""

	helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], START_TOKEN), END_TOKEN)
	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=projection_layer)
	decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
	decoder_outputs = decoder_outputs.rnn_output

	

	sentences = SENTENCES
	questions = SENTENCES

	"""#**Test Model**"""


	with open("/Users/Anurag/Downloads/gen-questions-master/doc_to_questions/output/generated_qa.csv", 'w') as file:
	    for i, batch in enumerate(training_data(sentences, questions)):
	        questions = session.run(decoder_outputs, {
	          encoder_inputs: batch["sentence_tokens"],

	          encoder_lengths: batch["sentence_lengths"]
	        })
	        if i > 99:
	            break

	        questions[:,:,UNKNOWN_TOKEN] = 0
	        questions = np.argmax(questions, 2)

	        for i in range(batch["size"]):
	            question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
	            s = batch["sentence_text"][i]
	            print("Sentence: " + s)
	            q = "".join(look_up_token(token) for token in question)
	            print("Question generated: " + q)
	            file.write(str(q))
	            file.write("\t")
	            file.write(str(s))
	            file.write("\n")
	    file.close()
	# for i, batch in enumerate(training_data(sentences, questions)):
	#     questions = session.run(decoder_outputs, {
	#       encoder_inputs: batch["sentence_tokens"],
	        
	#       encoder_lengths: batch["sentence_lengths"]
	#     })
	#     if i > 99:
	#         break

	#     questions[:,:,UNKNOWN_TOKEN] = 0
	#     questions = np.argmax(questions, 2)

	#     for i in range(batch["size"]):
	#         question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
	#         print("Sentence: " + str(i+1)+' '+ batch["sentence_text"][i])
	#         print("Question generated: " + "".join(look_up_token(token) for token in question))
	#         print()

