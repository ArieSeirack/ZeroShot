from transformers import AutoTokenizer, AutoModel
from utils import load_model_on_gpus

tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=True, revision="v1.1.0")

model = load_model_on_gpus("chatglm-6b", num_gpus=2)
model = model.eval()

print(1)

import os

root = "./datasets"
with open(os.path.join(root, 'classes.txt'), 'r') as f:
    object_categories = f.readlines()
object_categories = [i.strip() for i in object_categories]
classes = ' '.join(object_categories)
print(classes)
print(2)

collected_data = {}
nsent_percls = 30
n_more = 5

for i, n in enumerate(object_categories):
    response1, history = model.chat(tokenizer, f"Describe the possible shape, color, texture, or any other visual features about {n}. Generate {nsent_percls} distinct and concise examples in English. The length of every example should be less than fifty words", history=[])
    
    #response2, history = model.chat(tokenizer, f"Generate {n_more} more sentences about what other things that {n} is often accompanied by. Use English. The format should be: {n} is accompanied by (the names of the things). Filter out the things that are not in the {classes}.", history=[])    
    print(response1)
    #print(response2)
    collected_data[i] = response1.split('\n')


'''
for i, n in enumerate(object_categories):
    for j, n2 in enumerate(object_categories):
        response, history =  model.chat(tokenizer, f"{n} and {n2} appear together in what scenario? Give me one example in English. If they have no relationship, just tell me no. Your responese length must be less than 10 words.", history=[])
        print(response)
        collected_data[i+j] = response.split('\n')
'''

print(3)

import json

with open('ChatGLM_w2s_coco_10s_test8.json', 'w') as f:
    json.dump(collected_data, f, indent=4)

print(4)