from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the Mistral model
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",  # Adjust as needed for your setup
    torch_dtype=torch.float16  # Using half-precision for GPU
)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data['prompt']
        n_return_sequences = data['number_responses']
        #print("a7eh1")
        # Create messages format for Mistral model
       # print("a7eh2")
        # Apply chat template and encode
        encoded_inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
        #print("a7eh2.1")
        input_ids = encoded_inputs.to(model.device)
       # print("a7eh2.2")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)
      #  print("a7eh3")
        # Generate responses
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,  # Adjust as needed
            temperature=1,
            top_k=50,  # Adjust as needed
            num_return_sequences=n_return_sequences,
            do_sample=True  # Enable sampling
        )
     #   print("a7eh4")
        responses = []
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
           # print("a7eh5")
        return jsonify({'responses': responses})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

