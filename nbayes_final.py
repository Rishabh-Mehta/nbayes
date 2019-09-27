
import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt

random.seed(10)
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()

# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]
# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)
    
# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0

    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)

pspam =0
pham =0
spam_words = []
ham_words = []
#alpha = 0.1
for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
        pspam = pspam +1
    else:
        ham_words += words
        pham = pham + 1
        
input_words = spam_words + ham_words  # all words in the input vocabulary
pspam = pspam /(pspam+pham)
pham = 1 - pspam
print("Probablity of Spam",pspam)
print("Probablity of ham",pham)
print("---------------------------------------------------------")

# Count spam and ham occurances for each word
spam_counts = {}; ham_counts = {}
arr = []
I = []
arr.append(0.1) 
for i in range(-5,1):
    arr.append(2**i)
    I.append(i)
#print(word_spam_count)
# Spamcounts
p = []
f = []
r = []
A = []
def predict(alpha):
    
    print("Alpha = ",alpha)
    for word in spam_words:
        try:
            word_spam_count = spam_counts.get(word)
            spam_counts[word] = word_spam_count + 1
        except:
            spam_counts[word] = 1 + alpha  # smoothening

    for word in ham_words:
        try:
            word_ham_count = ham_counts.get(word)
            ham_counts[word] = word_ham_count + 1
        except:
            ham_counts[word] = 1 + alpha  # smoothening
            

    num_spam = len(spam_words)
    num_ham = len(ham_words)


    
    #Probablity of word is spam P(word|spam)= frequency of word in spam category / total number of words in spam 
    #training data 

    spam_probablity = spam_counts
    for key in spam_probablity:
        spam_probablity[key] = ((spam_probablity[key] ) / (num_spam + alpha * 20000))
    #Probablity of word is ham P(word|ham)= frequency of word in ham category / total number of words in ham 
    #training data

    ham_probablity = ham_counts
    for key in ham_probablity:
    
        ham_probablity[key] = ((ham_counts[key] ) / (num_ham + alpha * 20000))
    # Predictiction for test documents

    predict_label = []
    for ii in range(len(test_words)):
        words = test_words[ii]
        sp =pspam
        hp =pham
        for j in range(len(words)):
            word = words[j]
            b = ham_probablity.get(word,(alpha/(num_ham + alpha * 20000)))
            a = spam_probablity.get(word,(alpha/(num_spam + alpha * 20000)))
            sp = sp * a
            hp = hp * b
            
    # print(sp)
    # print(hp)
        if (sp > hp):
            predict_label.append(1)
        else:
            predict_label.append(0)
    # print("predict ",predict_label[ii])

    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for i in range(len(predict_label)):
        if(test_labels[i] ==1 and predict_label[i] ==1):
            true_positive = true_positive + 1
        elif(test_labels[i] ==0 and predict_label[i] ==0):
            true_negative = true_negative + 1
        elif(test_labels[i] ==0 and predict_label[i] ==1):
            false_positive = false_positive + 1
        elif(test_labels[i] ==1 and predict_label[i] ==0):
            false_negative = false_negative + 1
           
    print("Confusion Matrix")
    print(str(true_positive)+" "+str(false_positive))
    #print(true_negative)
    #print(false_positive)
    print(str(false_negative)+" "+str(true_negative))   
    
    precision = true_positive/(true_positive+false_positive)
    p.append(precision)
    print("Precision",precision)
    recall = true_positive/(true_positive+false_negative)
    r.append(recall)
    print("Recall",recall)
    fscore = (2*precision*recall)/(precision+recall)
    print(type(fscore))
    f.append(fscore)
    print("Fscore",fscore)
    accuracy= (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    A.append(accuracy)
    print("Accuracy",round(accuracy*100,3))
    print("---------------------------------------------------------")

for k in arr:
    predict(k)
    
print(p)
print(r)
print(f)

print(A)
plt.plot(arr,A,label='Accuracy')
plt.plot(arr,f,label='Fscore')
#plt.plot(arr,p,label='Precision')
#plt.plot(arr,r,label='Recall')
plt.xlabel('Alpha')
plt.legend()
plt.show()




    