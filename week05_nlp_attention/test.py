import collections
import jieba


def count_str(s: str = "I love love you you too"):
    return collections.Counter(s.split())


counter = count_str()

print(counter)
print(counter.items())
for item in counter.items():
    print(f"{item[0]} : {item[1]}")

sorted_items = sorted(counter.items(),
                      key=lambda x: x[1],
                      reverse=True)
print(sorted_items)

text = "我喜欢学习机器学习"
tokens = jieba.lcut(text)
print(tokens)

names = ['Tom', 'Jack', 'Lucy']
scores = [90, 85, 95]

result = list(zip(names, scores))
dictionary = dict(zip(names, scores))
print(result)
print(dictionary)

# 最常见用法：同时遍历两个列表
for name, score in zip(names, scores):
    print(f"{name} : {score}")

data = [
    ('Tom', 90),
    ('Jack', 85),
    ('Lucy', 95)
]
names_z, scores_z = zip(*data)
print(names_z)
print(scores_z)



