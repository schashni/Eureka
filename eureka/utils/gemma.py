from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class GemmaPrompt:
	def __init__(self, system_message):
		self.system_message = system_message
		self.turns = []

	def add_turn(self, role, message):
		assert (role == 'user' or role == 'model')
		self.turns.append({'role': role, 'message': message})

	def update_last_turn(self, role, message):
		assert len(self.turns) != 0

		reversed_list = reversed(self.turns)
		for index, turn in enumerate(reversed_list):
			if turn['role'] == role:
				reversed_list[index]['message'] = message
				break

		self.turns = reversed(reversed_list)

	def generate_turn(self, role, message):
		start = '<start_of_turn>'
		end = '<end_of_turn>'
		
		return start + role + ' ' + message + end

	def get_string(self):
		prompt = self.generate_turn('user', self.system_message)
		for turn in self.turns:
			prompt += self.generate_turn(turn.role, turn.message)
		return prompt		


access_token='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'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", bos_token = '', eos_token = '', token=access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", token=access_token, device_map="auto")

#input_text = GemmaPrompt('Who are you?').get_string()
input_text = '<start_of_turn>user Who is the first russian president?<end_of_turn><start_of_turn>model The first president of the Russian Federation was Vladimir Putin, who served from 2000 to 2008.<end_of_turn><start_of_turn>user Who was the president after him?<end_of_turn><start_of_turn>model '
#input_text = '<start_of_turn>user Who was the president after him?<end_of_turn><start_of_turn>model '
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

print(model.device)
start_time = time.time()
outputs = model.generate(**input_ids, max_new_tokens=150)
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time) 
print(tokenizer.decode(outputs[0]))