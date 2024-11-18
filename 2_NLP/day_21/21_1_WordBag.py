import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. 准备数据
data = {
    'text': [
        'I love programming',
        'Python is a fantastic programming language',
        'I enjoy hiking and outdoor adventures',
        'The weather is nice for a day out',
        'I hate waiting in line',
        'Traffic is terrible today',
        'I love ice cream',
        'The movie was boring',
        'I am excited about the concert',
        'Food is amazing'
    ],
    'label': [
        'positive',
        'positive',
        'positive',
        'positive',
        'negative',
        'negative',
        'positive',
        'negative',
        'positive',
        'positive'
    ]
}

df = pd.DataFrame(data)

# 2. 数据预处理
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. 构建词袋模型
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 4. 训练分类模型
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 5. 进行预测
y_pred = model.predict(X_test_vectorized)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))