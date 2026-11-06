# Методы распознавания образов (ТПУ, магистратура, 3 семестр)

| Работа | Содержание |
|---|---|
| `svm_first_lab.ipynb`, `SVM_classification/` | Классификация методом опорных векторов (scikit-learn, Breast Cancer и др.), подбор ядер и параметров |
| `HMM/` | Распознавание изолированных слов (названия фруктов) по речи: MFCC-признаки + скрытые марковские модели (`hmmlearn`). Аудиоданные в `HMM/audio`, `train_audio`, `test_audio` |
| `adversarial_attacks_against_machine_learning_models.ipynb`, `Adversarial ML/` | Состязательные атаки на CNN (VGG19/ResNet50) с помощью Foolbox: FGSM, DeepFool, шумовые атаки |

## Запуск
```bash
pip install numpy scipy scikit-learn matplotlib hmmlearn python_speech_features tensorflow foolbox jupyter
cd HMM/lab2 && python main.py
```
