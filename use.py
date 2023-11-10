import torch

from train import BertClassifier
from train import preprocessing_for_bert

model_path = './best_model.pkl'
model = BertClassifier()
model.load_state_dict(torch.load(model_path, 'cpu'))

x_use = ['质量不错', '质量太差了']

test_inputs, test_masks = preprocessing_for_bert(x_use)

y_pred = model(x_use)
print(y_pred.argmax()==0)



