import numpy
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize
from csv import DictReader
import csv
import wordninja
def delete_tokens(text):
    text = text.replace('@@ ', '')
    text = text.replace('<at>', '')
    text = text.replace('<url>', '')
    text = text.replace('<number>', '')
    text = text.replace('<heart>', '')
    text = text.replace('<cont>', '')
    text = text.replace('<first_speaker> ', '')
    text = text.replace('<second_speaker> ', '')
    text = text.replace('<third_speaker> ', '')
    text = text.replace('<minor_speaker> ', '')
    text = text.replace('<at> ', '')
    text = text.replace('<url> ', '')
    text = text.replace('<number> ', '')
    text = text.replace('<heart> ', '')
    text = text.replace('<cont> ', '')
    return text


def split_ninja(phrase):
    clean_phrase = delete_tokens(phrase)
    tokens = word_tokenize(clean_phrase)
    phrase_list = []
    list_ninja = [wordninja.split(word) for word in tokens]
    for word in list_ninja:
            for item in word:
                if item.isdigit() :
                    continue
                phrase_list.append(item)
    return phrase_list


def nltk_processing(phrase):

    ninja_phrase = split_ninja(phrase)
    filtered_phrase = [word for word in ninja_phrase if not word in stop_words]

    return filtered_phrase

def word_vec(phrase):
    list_of_vectors = []
    list_of_words = nltk_processing(phrase)
    if len(list_of_words) == 0:
        list_of_vectors.append(numpy.random.rand(300))
    X = []
    X_prim = []
    for word in list_of_words:
        try:
            list_of_vectors.append(model[word])
        except KeyError:
            print("The word is not in dictionary")
            list_of_vectors.append(numpy.random.rand(300))

    for j in range(0, 300):
        summ = 0

        for item in list_of_vectors:
            summ += item[j]
        mean = summ/len(list_of_vectors)
        X_prim.append(mean)
    X.append(X_prim)
    return X

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
stop_words = set(stopwords.words("english"))
X_train_list_response = []
X_train_list_context = []
count = 0
with open('validation.txt') as f:
    reader = DictReader(f, delimiter='\t')
    for row in reader:

        X_train_list_response = word_vec(row['response'])
        X_train_list_context = word_vec(row['context'])
        X_train_context = numpy.asarray(X_train_list_context)
        X_train_response = numpy.asarray(X_train_list_response)
        X_train = numpy.concatenate((X_train_context, X_train_response), axis=1)

        with open("validationtf.csv", "a") as f:
            writer = csv.writer(f)
            for item in X_train:
                writer.writerow(item)

        count += 1
        print(count, "lines were processed")




