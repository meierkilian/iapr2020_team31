import pickle

path_classifier='symbol_classifier_V2.pk'
lda_model = pickle.load(open(path_classifier,'rb'))

with open('symbol_classifier_V2.pk', 'wb') as dataFile:
		pickle.dump(lda_model, dataFile)