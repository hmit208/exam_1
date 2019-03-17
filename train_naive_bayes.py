import re
import pickle
import numpy as np
import gensim
import time

data = []
stopwords = []
with open('stopwords.txt', 'r') as f:
    for line in f:
        stopwords.append(line.replace("\n", ""))
vocabSet = pickle.load(open("dumped_files/vocabSet_v2.p", "rb"))


def stopwordsFiltering(array, string):  # loai bo tu dung
    for i in array:
        string = string.replace(" " + i + " ", " ")
        string = string.replace(" " + i + ".", ".")
        string = string.replace(" " + i + ",", ",")
    return string


def prepareString(s):
    tempArr = []
    s = s.split("\n")
    for i in range(0, len(s)):
        s[i] = re.sub(r'https?:\/\/.*[\r\n]*', "", s[i], flags=re.MULTILINE)
        s[i] = re.sub("%|:|'|@|#|\$|\,|\"|\(|\)|&|\*|Nguồn.*|[0-9]|\/|\.|\“|’|;| - |\]|\[|\?", '', s[i])
        s[i] = re.sub("  |   | - ", " ", s[i])
        s[i] = s[i].lower()
        s[i] = s[i].split(" ")
        tempArr += s[i]
    return tempArr


def prepareData():
    preparedData = []
    for i in range(1, 14):
        with open("classify_data/train/" + str(i) + ".txt") as f:
            s = f.read()
            s = stopwordsFiltering(stopwords, s)
        s = prepareString(s)
        preparedData.insert(i, s)
    return preparedData


def makeVocabSet():
    with open("v.txt") as f:
        s = f.read()
        s = re.sub("\n", " ", s)
        s = s.lower()
        s = re.sub(r'https?://\S+', "", s, flags=re.MULTILINE)
        s = gensim.utils.simple_preprocess(s)
        unique_words = sorted(set(s))
        pickle.dump(unique_words, open("dumped_files/vocabSet_v2.p", "wb"))


def load_data(file_path):
    lines = []
    with open(file_path) as f:
        for line in f:
            line = line.lower()
            line = stopwordsFiltering(stopwords, line)
            line = re.sub(r'https?://\S+', "", line, flags=re.MULTILINE)
            line = gensim.utils.simple_preprocess(line)
            lines.append(line)
    return lines


def cal_idf():
    N = 13 * 2000
    vocabSet = pickle.load(open("dumped_files/vocabSet_v2.p", "rb"))


def wordToIdx(vocabSet):
    wordToIdxMap = {}
    for idx, word in enumerate(vocabSet):
        wordToIdxMap[word] = idx
    return wordToIdxMap


def vectorize_text(document, vocabSet, wordToIdxMap):
    vector = np.zeros(len(vocabSet))
    for word in document:
        if word in wordToIdxMap:
            vector[wordToIdxMap[word]] += 1
    return vector


def count_all_vecto(vecto_list, vocab_len):
    final_vecto = np.zeros(vocab_len)
    for vecto in vecto_list:
        final_vecto += vecto
    return final_vecto


def train(tf_vectors, vocab_len):
    train_dict = {}
    for label, vecto in tf_vectors.items():
        print("train class: ", label)
        final_vecto = (vecto + 1) / (vecto.sum() + vocab_len)
        train_dict[label] = final_vecto
    return train_dict


def classify_doc(document, train_dict_matrix, wordToIdxMap):
    vecto = vectorize_text(document, vocabSet, wordToIdxMap)
    probabilities = vecto.dot(np.log(train_dict_matrix.T))
    outcome = np.argmax(probabilities) + 1
    return outcome


def accuracy(vecto_y, label_vecto):
    acc = np.sum(vecto_y == label_vecto)
    return acc / vecto_y.shape[0]


def test():
    vocabSet = pickle.load(open("dumped_files/vocabSet_v2.p", "rb"))
    train_dict = pickle.load(open("dumped_files/train_dict.p", "rb"))
    wordToIdxMap = wordToIdx(vocabSet)

    train_dict_matrix = np.array([vecto for i, vecto in train_dict.items()])
    print(train_dict_matrix.shape)

    test_data = load_data("classify_data/test/data.txt")
    test_label = []
    with open("classify_data/test/label.txt", "r") as file:
        for line in file:
            test_label.append(int(line.strip()))

    predictions = []
    for step, document in enumerate(test_data):
        print(step)
        predictions.append(classify_doc(document, train_dict_matrix, wordToIdxMap))
    return accuracy(np.array(predictions), np.array(test_label))


def main():
    vocabSet = pickle.load(open("dumped_files/vocabSet_v2.p", "rb"))
    data = pickle.load(open("dumped_files/documents.p", "rb"))
    tf_vectors = {}
    x = time.time()
    count = 0
    wordToIdxMap = wordToIdx(vocabSet)
    for i in range(1, 14):
        temp = 0
        print("classssssssssssssssssssss", count)
        vecto_list = []
        for j in data[i][0]:
            print(temp)
            vecto_list.append(vectorize_text(j, vocabSet, wordToIdxMap))
            temp += 1
        vecto = count_all_vecto(vecto_list, len(vocabSet))
        tf_vectors[i] = vecto
        count += 1
    print(time.time() - x)
    # pickle.dump(tf_vectors, open("dumped_files/tf_vectors.p", "wb"))
    print(tf_vectors)

    train_dict = train(tf_vectors, len(vocabSet))
    pickle.dump(train_dict, open("dumped_files/train_dict.p", "wb"))
    print(train_dict)


acc = test()
print("acc: ", acc)
