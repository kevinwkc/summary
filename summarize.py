from transformers import T5Tokenizer
from onnxruntime import InferenceSession
from onnxt5 import GenerativeT5

tokenizer = T5Tokenizer.from_pretrained('t5-base')
decoder_sess = InferenceSession('model/t5-decoder-with-lm-head.onnx')
encoder_sess = InferenceSession('model/t5-encoder.onnx')
#tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)

def infer(s):
    r=r=generative_t5(s, 16, temperature=0.)[0]
    print(r)

infer('translate English to French: I truly believe workplace bullying is a crime.')
infer('cola sentence: I truly believe workplace bullying is a crime.')
infer('''summarize: The example we're going to use is a very simple "address book" application that can read and write people's contact details to and from a file. Each person in the address book has a name, an ID, an email address, and a contact phone number. How do you serialize and retrieve structured data like this? There are a few ways to solve this problem: Use Python pickling. This is the default approach since it's built into the language, but it doesn't deal well with schema evolution, and also doesn't work very well if you need to share data with applications written in C++ or Java.''')
