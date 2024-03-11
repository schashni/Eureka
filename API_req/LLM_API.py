from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the model (similar to your provided code)
model_id = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    use_auth_token="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"
)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data['prompt']
        
        n_return_sequences = data['number_responses']
    

        messages = [{"role": "user", "content": prompt}]
        # Add your generation logic here
        # ...
        # For example:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    

        inputs = tokenizer.encode(prompt,return_tensors="pt")


        outputs = model.generate(
            input_ids=inputs.to(model.device),
            max_new_tokens=1000,
            temperature=1,
            top_k=50,  # You can adjust this value
            num_return_sequences=n_return_sequences,
            do_sample=True  # Enable sampling
        )
        responses=[]
        
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output)
            responses.append(response)
        return jsonify({'responses': responses})
   
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
