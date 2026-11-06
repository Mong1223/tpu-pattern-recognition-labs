# Подключение библиотек
from __future__ import print_function
import warnings
import os
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
# Отключаем назойливие предупреждения
warnings.filterwarnings('ignore')

# Из-за разных звуковых частот все равно будет вылезать ошибка
# Надо поменять частоты кадров с 22 кГц на 16 кГц.


def extract_mfcc(full_audio_path):
    # Считывание аудио файла и его звуковой дорожки
    sample_rate, wave = wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave, samplerate=sample_rate)
    return mfcc_features


def buildDataSet(dir):
    # Отфильтровываем аудиофайлы wav в каталоге
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        # Разбивание строк на разделители
        tmp = fileName.split('.')[0]  # 10_1
        label = tmp.split('_')[1]  # 1
        # Извлечение Мел-кепстральных коэффициентов на аудио дорожке
        feature = extract_mfcc(dir + "\\" + fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset


# Обучение модели смеси Гауссовых распределений на Марковских скрытых слоях
def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 20

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_iter=5)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # получаем оптимальные параметры
        GMMHMM_Models[label] = model
    return GMMHMM_Models


def main():
    trainDir = r"Audio/train_audio"
    trainDataSet = buildDataSet(trainDir)
    print("Завершили подготовку данных для обучения")

    hmmModels = train_GMMHMM(trainDataSet)
    print("Завершили обучение моделей GMM_HMM для цифр от 0 до 9")

    # Матрица вероятностей перехода между состояниями
    print(hmmModels['1'].transmat_)

    testDir = r"Audio/test_audio"
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Для тестируемой метки", label, "предсказанное значение", predict)
        if predict == label:
            score_cnt += 1
    print("Процент правильных предсказаний равен %.2f" % (100.0 * score_cnt / len(testDataSet.keys())), "%")


if __name__ == "__main__":
    main()
