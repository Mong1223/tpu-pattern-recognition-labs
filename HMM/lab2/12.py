from keras.datasets import mnist
data = mnist.load_data()

train_data = data[0]
test_data = data[1]

trainX,trainy = train_data[0], train_data[1]
testX,testy = test_data[0], test_data[1]

# trX =trainX.reshape((60000, 28*28))
# tsX =testX.reshape((10000, 28*28))
trX =trainX
tsX =testX

# from sklearn import decomposition
# pca = decomposition.PCA(n_components=2)
# X_tr_reduced = pca.fit_transform(trX)
# X_ts_reduced = pca.fit_transform(tsX)

X_tr_reduced = trX
X_ts_reduced = tsX

def buildDataSet(x,y):
    # Filter out the wav audio files under the dir
    dataset = {}
    for i in range(y.shape[0]):
        label = y[i]
        feature = x[i]
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num,n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models

trainDataSet = buildDataSet(X_tr_reduced, trainy)
testDataSet = buildDataSet(X_ts_reduced, testy)

from hmmlearn import hmm
import numpy as np

hmmModels = train_GMMHMM(trainDataSet)
print("Finish training of the GMM_HMM models for digits 0-9")

# print(hmmModels['1'].transmat_)

score_cnt = 0
for label in testDataSet.keys():
    feature = testDataSet[label]
    scoreList = {}
    for model_label in hmmModels.keys():
        model = hmmModels[model_label]
        score = model.score(feature[0])
        scoreList[model_label] = score
    predict = max(scoreList, key=scoreList.get)
    print("Test on true label ", label, ": predict result label is ", predict)
    if predict == label:
        score_cnt+=1
print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")