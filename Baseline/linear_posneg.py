import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc 
from scipy import interp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

def tokenizer(text):
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

def extract_data_from_file(file_path, delimiter, text_index, label_index, pos_label, skip_first_line):
	x_data, y_data = [], []
	with open(file_path, 'r', encoding='utf-8', errors='ignore') as lines:
		skip = False
		if skip_first_line:
			skip = True
		for line in lines:
			data = line.strip().split(delimiter)
			if skip or len(data) == 1:
				skip = False
				continue
			text = data[text_index]
			label = 1 if data[label_index] == 'pos' else -1
			x_data.append(text)
			y_data.append(label)

	x_data = np.array(x_data)
	y_data = np.array(y_data)
	d = {'text': x_data, 'label': y_data}
	data = pd.DataFrame(data=d)
	return data

def gen_roc(pipe, X, y, method_label):
	kfold = StratifiedKFold(n_splits=5, shuffle=True)

	fig = plt.figure(figsize=(7,5))
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []
	i=0

	for train_ind, test_ind in kfold.split(X, y):
		# Fit then Train
		pipe.fit(X[train_ind], y[train_ind])
		probas = pipe.predict_proba(X[test_ind])

		# ROC
		fpr, tpr, thresholds = roc_curve(y[test_ind], probas[:, 1], pos_label=1)
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr,tpr)
		# plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %.2f)" % (i+1, roc_auc))
		i+=1

	# plt.plot([0,1], [0,1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
	mean_tpr /= i
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %.2f)' % mean_auc, lw=2)
	# plt.plot([0,0,1], [0,1,1], lw=2, linestyle=':', color='black', label='perfect performance')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.title('%s Receiver Operator Characteristic' % method_label)
	plt.legend(loc='lower right')
	plt.show() 

data_path = "/Users/Camille/data/NTCIR/data.txt"

data = extract_data_from_file(data_path, '\t', 4, 2, 'pos', True)

# train, test = train_test_split(data, test_size = 0.2)

pipe_svm = Pipeline([('scl', TfidfVectorizer(tokenizer=tokenizer)), ('svc', SVC(kernel='linear', C=1.0, random_state=0, probability=True))])

#Linear SVM
print("Linear SVM")
pipe_svm.set_params(svc__kernel='linear', svc__C=1.0, svc__random_state=0)
gen_roc(pipe_svm, data['text'], data['label'], "Linear SVM")



