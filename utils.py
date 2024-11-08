import torch
import torch.backends.cudnn 
import torch.cuda
import numpy as np
import random
import warnings
import gc

# ------------------------ CUDA MEMORY ----------------------------

def check_cuda_memory():
    '''If available, shows the current memory usage of each GPU.'''
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        for device in range(num_cuda_devices):
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            current_memory_usage = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"Current {torch.cuda.get_device_name(device)} memory usage: {current_memory_usage:.3f}/{total_memory:.3f} GiB")
    else:
        print("CUDA is not available.")

def empty_cuda_memory():
    '''If available, empties GPU memory.'''
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ------------------------ REPRODUCIBILITY ----------------------------

def set_deterministic_mode(SEED):
    '''Set seed for PRNGs on torch, random, and numpy. Set cuDNN operations as deterministic except for benchmark mode.'''
    torch.manual_seed(SEED)                       
    random.seed(SEED)                             
    np.random.seed(SEED)             
    torch.cuda.manual_seed_all(SEED)             
    torch.backends.cudnn.benchmark = False # choice of cuDNN algorithm should be deterministic for fixed input size and fixed model
    torch.backends.cudnn.deterministic = True # does not guarantee determinism if other dependencies are using non-deterministic operations

# ------------------------ PRE-PROCESSING ----------------------------

def tokenize(data, tokenizer, max_length, labeled=True):
    '''Returns tokenized inputs formatted w.r.t. the prompt template [Text: "x_i". \nLabel: y_i].'''

    batch_size = len(data['text'])
    inputs = [f'Text: "{text}' for text in data['text']] # input text to be passed to the LM
    if labeled:
      labels = [f'\nLabel: {label}' for label in data['label']]
    else:
      labels = [f'\nLabel: ' for label in data['label']]
    tokenized_inputs = tokenizer(inputs) # tokenized input text
    tokenized_labels = tokenizer(labels) # tokenized labels

    for i in range(batch_size):
        sample_input_ids = tokenized_inputs['input_ids'][i] # input ids of i-th sample
        label_input_ids = tokenized_labels['input_ids'][i] # label ids of i-th sample

        tokenized_inputs['input_ids'][i] = sample_input_ids + [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) # right padding
        tokenized_inputs['input_ids'][i] = tokenized_inputs['input_ids'][i][:max_length - len(label_input_ids) - 1] + [tokenizer('"')['input_ids'][1]] # truncation + right quote
        tokenized_inputs['input_ids'][i] = tokenized_inputs['input_ids'][i] + label_input_ids # adding label
        
        tokenized_labels['input_ids'][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids
        tokenized_inputs['labels'] = tokenized_labels['input_ids']

        tokenized_inputs['attention_mask'][i] = [1] * max_length
  
    return tokenized_inputs

def get_labels_from_dataloader(batch, tokenizer):
    '''Get the labels from a DataLoader batch.'''
    labels = list()
    for tensor in batch['labels']:
        label = int(tokenizer.decode(tensor[-1]))
        labels.append(label)
    return labels

# ------------------------ TRAINING ----------------------------

class EarlyStopper:
    '''Implements early stopping by returning counter (number of consecutive loss increases) and stop (boolean).'''
    def __init__(self, patience=1, delta=0):
        self.patience = patience # maximum number of loss increases before training is stopped
        self.delta = delta # minimum loss variation for the loss to be considered increased 
        self.counter = 0
        self.min_loss = float('inf')
        self.stop = False

    def __call__(self, loss):
        if loss < self.min_loss: # improvement
            self.counter = 0
            self.min_loss = loss
        elif loss >= (self.min_loss + self.delta): # no improvement
            self.counter += 1
            if self.counter == self.patience:
                self.stop = True
        return self.counter, self.stop

# ------------------------ POST-PROCESSING ----------------------------

def get_labels_from_predictions(predictions, expected_labels, tokenizer):
    '''Retrieve labels from outputs of forward() function – i.e. __call__() of transformers.AutoModel class. Since forward() implements
    teacher forcing and input prompts end with 'Label: <label>', predicted labels are tokens at index -2. If the label does not match
    any expected label, it generates a random label.'''
    labels = list()
    missed = 0
    for prediction in predictions:
        label = tokenizer.decode(prediction[-2]) # decoding token at index -2
        if label in [str(item) for item in expected_labels]:
            labels.append(int(label))
        else:
            labels.append(random.choice(expected_labels)) # if the token does not match any expected label, a random label is generated
            missed += 1
    if missed:
        warnings.warn(f'A number of {missed} labels have been randomly selected because generated labels did not match any expected label.', UserWarning)
    return labels


def get_labels_from_texts(texts, expected_labels):
    '''Retrieves labels from text outputs of model.generate(), assuming that the label is the last character. If the label does not match
    any expected label, it generates a random label.'''

    labels = list()
    missed = 0
    for text in texts:
        if text[-1] in [str(item) for item in expected_labels]:
            labels.append(int(text[-1]))
        else:
            labels.append(random.choice(expected_labels))
            missed += 1
    if missed:
        warnings.warn(f'A number of {missed} labels have been randomly selected because generated labels did not match any expected label.', UserWarning)
    return labels




        
        
    

                                         
        