from sklearn.feature_extraction.text import TfidfVectorizer

# 创建一个文本数据集
texts = [
    "这是一个示例文本。",
    "这是另一个示例文本。",
    "这是第三个示例文本。",
    "这是第四个示例文本，包含一些重复的词汇。"
]

# 创建一个 TfidfVectorizer 对象
vectorizer = TfidfVectorizer()

# 对文本数据进行向量化处理
tfidf_matrix = vectorizer.fit_transform(texts)

# 打印词汇表
print("词汇表:", vectorizer.get_feature_names_out())

# 打印 TF-IDF 矩阵
print("TF-IDF 矩阵:")
print(tfidf_matrix.toarray())

# 如果你想查看某个文档的 TF-IDF 值，可以使用 toarray() 方法
document_index = 0
print(f"文档 {document_index} 的 TF-IDF 值:")
print(tfidf_matrix.toarray()[document_index])
