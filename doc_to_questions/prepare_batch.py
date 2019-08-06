import numpy as np

def look_up_char(char):
  return char_to_idx.get(char, UNKNOWN_TOKEN)


def tokenize(string):
  return list(string)

def prepare_batch(batch):
  id_to_indices = {}
  sentence_id = []
  sentence_text = []
  sentence_chars = []
  for i, entry in enumerate(batch):
    id_to_indices.setdefault(entry["sentence_id"], []).append(i)
    sentence_id.append(entry["sentence_id"])
    sentence_text.append(entry["sentence_text"])
    sentence_chars.append(entry["sentence_chars"])
  batch_size_prepare = len(batch)
  max_sentence_length = max((len(sentence) for sentence in sentence_chars), default=0)
  
  sentence_tokens = np.zeros((batch_size_prepare, max_sentence_length), dtype = np.int32)
  sentence_lengths = np.zeros(batch_size_prepare, dtype = np.int32)
  
  for i in range(batch_size_prepare):
    for j, char in enumerate(sentence_chars[i]):
      sentence_tokens[i, j] = look_up_char(char)
    sentence_lengths[i] = len(sentence_chars[i])
    
  return {
      "size": batch_size_prepare,
      "sentence_id": sentence_id,
      "sentence_tokens": sentence_tokens,
      "sentence_lengths": sentence_lengths,
      "sentence_text": sentence_text
  }

def process_batch(sentence):
  batch = []
  for text in sentence.values():
    if len(batch) + len(text) > 1:
      yield prepare_batch(batch)
      batch = []
    batch.extend(text)
  if batch:
    yield prepare_batch(batch)

def read_data(sentences):
  _sentence = {}
  for i, sentence in enumerate(sentences):
    sentence_id = i
    existing = _sentence.setdefault(sentence_id, [])
    sentence_text = sentence
    sentence_chars = tokenize(sentence)
    existing.append({
        "sentence_id": sentence_id,
        "sentence_text": sentence_text,
        "sentence_chars": sentence_chars
    })
  return _sentence

def get_data(sentences):
  return process_batch(read_data(sentences))