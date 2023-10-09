import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {"ham": 0, "spam": 1} 
        self.num2label = {0: "ham", 1: "spam"} 
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        for i in range(len(self._x)):
            self._x[i] = re.sub(r"[^\w\s]", " ", self._x[i])
            self._x[i] = " ".join(self._x[i].split())
            self._x[i] = self._x[i].lower()
            
        self._y[self._y == "ham"] = 0
        self._y[self._y == "spam"] = 1


    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        indices = np.arange(0, len(self._x))
        np.random.shuffle(indices)
        
        val_indices = indices[:round(val * len(self._x))]
        test_indices = indices[round(val * len(self._x)):round((val + test) * len(self._x))]
        train_indices = indices[round((val + test) * len(self._x)):]
        
        X_val = np.array([self._x[index] for index in val_indices])
        y_val = np.array([self._y[index] for index in val_indices])
        X_test = np.array([self._x[index] for index in test_indices])
        y_test = np.array([self._y[index] for index in test_indices])
        X_train = np.array([self._x[index] for index in train_indices])
        y_train = np.array([self._y[index] for index in train_indices])
        
        self.val = (X_val, y_val)
        self.test = (X_test, y_test)
        self.train = (X_train, y_train)
