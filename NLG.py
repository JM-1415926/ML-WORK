import nltk
import jieba  # 中文分词工具
import string

n_param = 2

from nltk.util import pad_sequence
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

# 停用词列表（可以根据需要扩展）
stop_words = set(['的', '了', '是', '在', '和', '也', '我', '有', '就', '不', '都', '而', '一个', '上', '下', '着', '这', '那', '你', '他', '她'])

# 读取中文数据集
document = open("dataset.txt", encoding="ANSI").read()

# 去除标点符号
# string.punctuation 只包含西文标点，因此需要自己定义中文标点
punctuation = string.punctuation + '，。、；！？【】（）《》“”‘’：… '
document_no_punc = "".join([char for char in document if char not in punctuation])

# 使用 jieba 进行分词
seg_list = jieba.cut(document_no_punc, cut_all=False)
# 去除停用词
tokenized = [word for word in seg_list if word not in stop_words]

# 为 n-gram 模型生成训练语料和词汇表
corpus, vocab = padded_everygram_pipeline(n_param, [tokenized])

# 创建和训练 Laplace n-gram 模型
lm = Laplace(n_param)
lm.fit(corpus, vocab)

# 打印词汇表大小
print("Size of vocabulary:", str(len(lm.vocab)))

# 使用生成器生成文本
generated_text = lm.generate(40, text_seed=['笑'], random_seed=33)
print("Generated text:", ''.join(generated_text))
