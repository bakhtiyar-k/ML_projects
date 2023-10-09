import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None 
        self.num2label = None 
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", чтобы заполнить все атрибуты данного класса.
        '''

        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        self._train_X, self._train_y = dataset.train[0], dataset.train[1]
        self._val_X, self._val_y = dataset.val[0], dataset.val[1]
        self._test_X, self._test_y = dataset.test[0], dataset.test[1]
        
        for msg, label in zip(dataset.train[0], dataset.train[1]):
            
            if label == 0:
                
                for word in msg.split():
                    self.vocab.add(word)
                    if word not in self.ham:
                        self.ham[word] = 0
                    else:
                        self.ham[word] += 1
            else:
                
                for word in msg.split():
                    self.vocab.add(word)
                    if word not in self.spam:
                        self.spam[word] = 0
                    else:
                        self.spam[word] += 1
                        
        self.Nvoc = len(self.vocab) 
        self.Nspam = len(self.spam)
        self.Nham = len(self.ham)

    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''

        p_spam = len(self._train_y[self._train_y == 1]) / len(self._train_y)
        p_ham = len(self._train_y[self._train_y == 0]) / len(self._train_y)
        
        pspam = 1
        pham = 1
        words = re.sub(r"[^\w\s]", " ", message).lower().split()
                
        for word in words:
            if word in self.spam:
                pspam *= ((self.spam[word] + self.alpha) / (sum(self.spam.values()) + self.alpha * self.Nvoc))
            else:
                pspam *= ((0 + self.alpha) / (sum(self.spam.values()) + self.alpha * self.Nvoc))
                
        for word in words:
            if word in self.ham:
                pham *= ((self.ham[word] + self.alpha) / (sum(self.ham.values()) + self.alpha * self.Nvoc))
            else:
                pham *= ((0 + self.alpha) / (sum(self.ham.values()) + self.alpha * self.Nvoc))
        
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        '''
        count_of_corrects = 0
        
        for message, true_label in zip(self._val_X, self._val_y):
            if self.label2num[self.inference(message)] == true_label:
                count_of_corrects += 1
        
        val_acc = count_of_corrects / len(self._val_y) * 100

        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        '''
        count_of_corrects = 0
        
        for message, true_label in zip(self._train_X, self._train_y):
            if self.label2num[self.inference(message)] == true_label:
                count_of_corrects += 1
        
        test_acc = count_of_corrects / len(self._train_y) * 100

        return test_acc


