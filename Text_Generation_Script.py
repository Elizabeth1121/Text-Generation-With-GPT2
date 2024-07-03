from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode some text to generate from
#input_text = "As a junior developer, I want to learn more about"
input_text = "As a medical student, I want to learn more about"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
#output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Generate Text V2
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,  # Prevents the model from repeating the same n-grams
    temperature=0.7,         # Controls the randomness of the generation
    top_k=50,                # Narrows down the possible next words at each step
    top_p=0.95,              # Chooses from the top p probability distribution
    eos_token_id=tokenizer.eos_token_id,  # End of sequence token
    pad_token_id=tokenizer.eos_token_id   # Padding token
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)