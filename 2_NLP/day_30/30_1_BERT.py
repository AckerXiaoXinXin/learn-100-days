import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


data = load_dataset('imdb')


model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def process_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


encoded_dataset = data.map(process_function, batched=True)

train_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
)

trainer.train()

evaluation_result = trainer.evaluate()
print(f'Evaluation result: {evaluation_result}')


def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions


sample_text = "This movie was fantastic! I really enjoyed it."
prediction = predict(sample_text)
print(f'Prediction for "{sample_text}": {prediction}')

