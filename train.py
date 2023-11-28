import os
import torch
import numpy as np
import random
import datetime
import timm
import json
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, StepLR
from PyQt6 import QtCore


class MyNet(torch.nn.Module):
    # Класс конструктор нейросетевой модели классификации изображений сигналов
    def __init__(self, model_name='resnet50', pretrained=False, num_classes=10):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        # Замораживаем веса, чтобы не использовать лишние веса в обучении, а обучать только последний слой
        for name, param in self.model.named_parameters():  # Проходим по параметрам модели (каждый параметр - это каждый слой, model.parameters нам отдаст некоторый итератор по слоям)
            # if 'layer1' in name and model_name is 'yamnet': continue # Пропускаем первый слой модели
            param.requires_grad = False  # Для каждого параметра и слоя:"requires grad = False", то есть уже не требуется вычисление градиента для данного слоя. И получается, что у нас вся сетка будет заморожена, то есть мы не сможем вообще ничего обучать

        # Добавляем полносвязанную классифицирующую голову
        if 'efficientnet' in model_name:  # Для EfficientNet есть небольшие отличия в названии слоев
            fc_inputs = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.classifier = nn.Linear(fc_inputs, num_classes)
        else:
            fc_inputs = self.model.fc.in_features
            self.model.fc = nn.Linear(fc_inputs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def seed_everything(seed):
    # Фиксация seed для воспроизводимости
    random.seed(seed)  # фиксируем генератор случайных чисел
    os.environ['PYTHONHASHSEED'] = str(seed)  # фиксируем заполнения хешей
    np.random.seed(seed)  # фиксируем генератор случайных чисел numpy
    torch.manual_seed(seed)  # фиксируем генератор случайных чисел pytorch
    torch.cuda.manual_seed(seed)  # фиксируем генератор случайных чисел для GPU


class TrainingModelThread(QtCore.QThread):
    # Класс потока для обучения модели
    _progress_signal = QtCore.pyqtSignal(int)  # Сигнал для отправки значения текущего шага обучения
    _data_len_signal = QtCore.pyqtSignal(int)  # Сигнал для отправки значения длины текущего загрузчика набора данных
    _log_signal = QtCore.pyqtSignal(str)  # Сигнал для отправки сообщения о текущем этапе обучения
    _progress_msg_signal = QtCore.pyqtSignal(str)  # Сигнал для отправки сообщения о качестве модели на текущем этапе
                                                   # обучения и сообщения о сохранении модели

    def __init__(self,
                 model_name,    # Наименование модели 'resnet50', 'resnext101_32x8d', 'inception_v3', 'densenet169'и т.п., str
                 model_dir,     # Путь до директории, где будет создана директория данной модели, str
                 classes,       # Список наименований классов, например, ["Q3N#3", "-1", "Q3N#2", "Q3N#1", "P0N#2", "P0N#1"], list(str,..)
                 lr,            # Начальная скорость обучения, float
                 scheduler_name,# Наименование функции изменения скорости, str
                 epochs,        # Кол-во эпох обучения, int
                 train_loader,  # Загрузчик обуч. данных, torch.utils.data.DataLoader
                 test_loader,   # Загрузчик тест. данных, torch.utils.data.DataLoader
                 image_size,    # Размер изображения, int
                 batch_size):   # Размер батча, int
        super(TrainingModelThread, self).__init__()
        self.model_name = model_name
        self.model_dir = model_dir
        self.classes = classes
        self.num_classes = len(classes)
        self.lr = lr
        self.scheduler_name = scheduler_name
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_size = image_size
        self.batch_size = batch_size

        self.model = None     # Обучаемая модель
        self.optimizer = None # Функция оптимизации модели
        self.device = None    # Девайс, на котором происходят расчеты (процессор/видеокарта)
        self.loss = None      # Функция ошибки
        self.scheduler = None # Функция изменения скорости обучения

    def __del__(self):
        self.wait()

    def run(self):
        # Запуск потока
        self.start_train()

    def train_val_model(self):
        # Функция обучения и инференса модели
        # Создание папки для сохранения модели и логов
        date_now = datetime.datetime.now()
        SAVE_DIR = self.model_dir + '\\' + f"{self.model_name}_{date_now.day}{date_now.month}{date_now.year}_{date_now.hour}"
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        BEST_MPODEL_NAME = f'{self.model_name}_best.pth'
        # Инициализация словаря с описанием модели
        descr = {"model_name": self.model_name,
                 "model_path": SAVE_DIR + '\\' + BEST_MPODEL_NAME,
                 "classes": {cls: i for i, cls in enumerate(self.classes)},
                 "img_size": self.image_size}

        loss_hist = {'train': [], 'val': []}
        acc_hist = {'train': [], 'val': []}

        best_acc = 0.
        seed_everything(1027)

        for epoch in range(self.epochs):
            for phase, ru_phase in zip(['train', 'val'], ["Обучение", "Тестирование"]):
                if phase == 'train':  # Если фаза == train
                    dataloader = self.train_loader  # Берем train_dataset
                    self.model.train()  # Модель в training mode - обучение
                else:  # Если фаза == val
                    dataloader = self.test_loader  # Берем valid_dataset
                    self.model.eval()  # Модель в evaluate mode - валидация (Фиксируем модель, иначе у нас могут изменяться параметры слоя батч-нормализации и изменится нейронка с течением времени)
                # Отправление сообщения о текущем этапе и эпохе обучения
                self._log_signal.emit("{}, эпоха {}/{}:".format(ru_phase, epoch + 1, self.epochs))
                self._data_len_signal.emit(len(dataloader))
                running_loss = 0.
                running_acc = 0.
                # Итерируемся по dataloader
                for i, data in enumerate(dataloader):
                    self._progress_signal.emit(i + 1) # Отправляется сообщение о текущем этапе обучения для progress bar
                    inputs = data['images'].to(self.device)  # Тензор с изображениями переводим на GPU
                    labels = data['labels'].to(self.device)  # Тензор с лейблами переводим на GPU
                    self.optimizer.zero_grad()  # Обнуляем градиент,чтобы он не накапливался
                    with torch.set_grad_enabled(phase == 'train'):  # Если фаза train, то активируем все
                        # градиенты (те, которые не заморожены)
                        preds = self.model(inputs)  # Считаем предикты, input передаем в модель
                        loss_value = self.loss(preds, labels)  # Посчитали  Loss
                        preds_class = preds.argmax(dim=1)  # Получаем класс, берем .argmax(dim=1) нейрон с максимальной активацией
                        if phase == 'train':
                            loss_value.backward()  # Считаем градиент
                            self.optimizer.step()  # Считаем шаг градиентного спуска
                    # Статистика
                    running_loss += loss_value.item()  # Считаем Loss
                    running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()  # Считаем accuracy

                epoch_loss = running_loss / len(dataloader)  # Loss'ы делим на кол-во бачей в эпохе
                epoch_acc = running_acc / len(dataloader)  # Считаем Loss на кол-во бачей в эпохе
                # Сообщение с значением функции ошибки и качества модели на текущем этапе
                progress_msg = ("{} значения функции ошибки: {:.2f} "
                                "точность: {:.2f}").format(ru_phase, epoch_loss, epoch_acc)
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # Если качество модели улучшилось, сохраняем её и добавляем об этом событии запись
                    progress_msg += f'\nЭпоха {epoch + 1} -Сохранение модели с лучшей точностью: {best_acc*100:.2f} %\n'
                    torch.save(self.model.state_dict(), descr["model_path"])
                self._progress_msg_signal.emit(progress_msg) # Отправка сообщения в сигнал
                loss_hist[phase].append(epoch_loss)
                acc_hist[phase].append(epoch_acc)

        descr["acc"] = best_acc
        # Сохранение файла описания модели в файл description.json
        with open(SAVE_DIR + '/' + 'description.json', 'w') as f:
            json.dump(descr, f)
        return loss_hist, acc_hist

    # Определяем разные типы функции изменения скорости обучения
    def get_scheduler(self):
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=4, verbose=False,
                                               eps=1e-6)
        elif self.scheduler_name == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=16, gamma=0.1)
        elif self.scheduler_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=6, eta_min=lr * 1e-2, last_epoch=-1)
        elif self.scheduler_name == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=6, T_mult=1, eta_min=lr * 1e-2,
                                                         last_epoch=-1)

    def start_train(self):
        # Формат pretrained=True - нам нужны веса, которые получились вследствие обучения этой модели на датасете ImageNet
        self.model = MyNet(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Определяем Loss функцию
        # В данном случае - это кросс-энтропия
        self.loss = nn.CrossEntropyLoss()
        # Метод градиентного спуска AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.get_scheduler()
        loss, acc = self.train_val_model()
