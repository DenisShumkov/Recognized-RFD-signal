# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets, QtCore, QtGui
import dataset
import os
import sys
import train
import torch


class TrainWindow(QtWidgets.QWidget):
    "Класс для вывода прогресса обучения НС"
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.resize(300, 100)
        self.setWindowTitle("Процесс обучения...")
        self.train_progress = QtWidgets.QProgressBar(self)
        self.log_label = QtWidgets.QLabel("")
        self.progress_label = QtWidgets.QLabel("")
        self.close_button = QtWidgets.QPushButton("ОК")
        self.close_button.setDisabled(True)
        self.close_button.clicked.connect(self.close)

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.log_label)
        self.vbox.addWidget(self.train_progress)
        self.vbox.addWidget(self.progress_label)
        self.vbox.addWidget(self.close_button)
        self.setLayout(self.vbox)

    def progressbar_accept(self, msg):
        # Обновление значения в progress bar
        self.train_progress.setValue(msg)

    def set_len_progressbar(self, msg):
        # Установка длины progress bar
        self.train_progress.setRange(0, msg)

    def log_accept(self, msg):
        # Вывод лог сообщения в окно прогресса обучения
        self.log_label.setText(msg)

    def progress_msg_accept(self, msg):
        # Вывод сообщения о текущем этапе обучения
        self.progress_label.setText(msg)


class StartWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Обучение нейронной сети распознавания сигналов")
        self.resize(500, 100)

        self.vbox = QtWidgets.QVBoxLayout()
        # Инициализация поля для ввода директории набора данных
        self.datasetpath_line = QtWidgets.QLineEdit("H:\SchebnevRadar3.5GHz\Radar3_5GHzSpectrograms")
        self.datasetpath_form = QtWidgets.QFormLayout()
        self.datasetpath_form.addRow("&Директория набора данных:", self.datasetpath_line)
        self.vbox.addLayout(self.datasetpath_form)

        # Инициализация поля для выбора размера входного изображения
        self.inputsize_box = QtWidgets.QComboBox()
        self.inputsize_box.addItems([str(pow(2, x)) + "x" + str(pow(2, x)) for x in range(5, 11)])
        self.inputsize_box.setEditable(False)
        self.inputsize_box.setCurrentText("512x512")
        self.inputsize_form = QtWidgets.QFormLayout()
        self.inputsize_form.addRow("&Размер входного изображения:", self.inputsize_box)
        self.vbox.addLayout(self.inputsize_form)

        # Инициализация поля для выбора размера батча изображений (размер = 2^n)
        self.batch_line = QtWidgets.QSlider()
        self.batch_line.setRange(0, 6)
        self.batch_orient = QtWidgets.QStyleOptionSlider()
        self.batch_line.setOrientation(self.batch_orient.orientation.Horizontal)
        self.batch_form = QtWidgets.QFormLayout()
        self.batch_label = QtWidgets.QLabel(f"Размер батча: {pow(2, self.batch_line.value())}")
        self.batch_form.addRow(self.batch_label, self.batch_line)
        self.batch_line.valueChanged.connect(self.batch_slider_chng)
        self.vbox.addLayout(self.batch_form)

        # Инициализируем кнопку для загрузки набора данных
        self.loaddataset_button = QtWidgets.QPushButton("Загрузить набор данных")
        self.vbox.addWidget(self.loaddataset_button)
        self.loaddataset_button.clicked.connect(self.on_clicked_loaddataset_button)

        # Инициализируем поле вывода классов набора данных и скрываем его
        self.datasetdiscr_text = QtWidgets.QTextEdit()
        self.datasetdiscr_text.hide()
        self.datasetdiscr_text.setReadOnly(True)
        self.vbox.addWidget(self.datasetdiscr_text)

        # Инициализация поля для выбора архитектуры нейронной сети
        self.modelname_box = QtWidgets.QComboBox()
        self.modelname_box.addItems(['resnet50', 'resnext101_32x8d', 'inception_v3', 'densenet169', 'efficientnet_b0'])
        self.modelname_box.setEditable(False)
        self.modelname_form = QtWidgets.QFormLayout()
        self.modelname_form.addRow("&Архитектура нейронной сети:", self.modelname_box)
        self.vbox.addLayout(self.modelname_form)

        # Инициализация поля для ввода количества эпох
        self.epochs_line = QtWidgets.QLineEdit()
        validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression("[0-9]*"))  # Только целые числа
        self.epochs_line.setValidator(validator)
        self.epochs_form = QtWidgets.QFormLayout()
        self.epochs_form.addRow("&Количество эпох обучения:", self.epochs_line)
        self.vbox.addLayout(self.epochs_form)

        # Инициализация поля для ввода начальной скорости обучения
        self.lr_line = QtWidgets.QSlider()
        self.lr_line.setMinimum(-12)
        self.lr_line.setMaximum(-1)
        # self.lr_orient = QtWidgets.QStyleOptionSlider()
        # self.lr_line.setOrientation(self.lr_orient.orientation.Horizontal)
        self.lr_line.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.lr_form = QtWidgets.QFormLayout()
        self.lr_label = QtWidgets.QLabel(f"Начальная скорость обучения: 1e{self.lr_line.value()}")
        self.lr_form.addRow(self.lr_label, self.lr_line)
        self.lr_line.valueChanged.connect(self.lr_slider_chng)
        self.vbox.addLayout(self.lr_form)

        # Инициализация поля для выбора функции изменения скорости обучения
        self.scheduler_box = QtWidgets.QComboBox()
        self.scheduler_box.addItems(['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'])
        self.scheduler_box.setEditable(False)
        self.scheduler_form = QtWidgets.QFormLayout()
        self.scheduler_form.addRow("&Метод изменения скорости обучения:", self.scheduler_box)
        self.vbox.addLayout(self.scheduler_form)

        # Инициализация поля для ввода директории, где сохранится модель
        self.modeldir_line = QtWidgets.QLineEdit("H:\SchebnevRadar3.5GHz")
        self.modeldir_form = QtWidgets.QFormLayout()
        self.modeldir_form.addRow("&Директория обученной модели:", self.modeldir_line)
        self.vbox.addLayout(self.modeldir_form)

        # Кнопка начала обучения
        self.starttraining_button = QtWidgets.QPushButton("Начать обучение модели")
        self.vbox.addWidget(self.starttraining_button)
        self.starttraining_button.setDisabled(True)
        self.starttraining_button.clicked.connect(self.on_clicked_strttrain_button)

        self.setLayout(self.vbox)

        self.train_progress_window = TrainWindow()  # Окно процесса обучения НС
        self.train_thread = None                    # Поток обучения НС
        self.train_loader = None                    # Загрузчик обуч. набора данных
        self.test_loader = None                     # Загрузчик тест. набора данных
        self.image_size = None                      # Размер изображения
        self.classes = None                         # Список классов в наборе данных
        self.batch_size = None                      # Размер батча

    def on_clicked_loaddataset_button(self):
        self.image_size = int(
            self.inputsize_box.currentText().split("x")[0])  # Записываем значение размера изображения в переменную
        self.batch_size = pow(2, self.batch_line.value())  # Записываем значение размера батча в переменную
        train_dataset = dataset.MakeDataset(self.datasetpath_line.text(), self.image_size, "train")
        test_dataset = dataset.MakeDataset(self.datasetpath_line.text(), self.image_size, "test")
        self.classes = train_dataset.classes
        # Выводим в поле описание набора данных
        self.datasetdiscr_text.setText(f"Классы набора данных:\n {', '.join(train_dataset.classes)}\n"
                                       f"Набор данных                           Кол-во изображений\n"
                                       f"----------------------------------------------------------\n"
                                       f"Обучающий набор данных     {len(train_dataset)}\n"
                                       f"Тестовый набор данных           {len(test_dataset)}")
        self.datasetdiscr_text.show()
        self.starttraining_button.setDisabled(False) # Активируем кнопку запуска обучения

        # Инициализируем загрузчки наборов данных
        def collate_fn(batch):
            return {
                'images': torch.stack([x[0] for x in batch]),
                'labels': torch.tensor([x[1] for x in batch]),
            }

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, collate_fn=collate_fn)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                       shuffle=True, collate_fn=collate_fn)

    def on_clicked_strttrain_button(self):
        self.train_progress_window.show()
        self.train_thread = train.TrainingModelThread(self.modelname_box.currentText(), self.modeldir_line.text(),
                                                      self.classes, pow(10, self.lr_line.value()),
                                                      self.scheduler_box.currentText(), int(self.epochs_line.text()),
                                                      self.train_loader, self.test_loader, self.image_size,
                                                      self.batch_size)
        self.train_thread._progress_signal.connect(self.train_progress_window.progressbar_accept)
        self.train_thread._log_signal.connect(self.train_progress_window.log_accept)
        self.train_thread._progress_msg_signal.connect(self.train_progress_window.progress_msg_accept)
        self.train_thread._data_len_signal.connect(self.train_progress_window.set_len_progressbar)
        self.train_thread.start()
        self.train_progress_window.close_button.setDisabled(False)

    def lr_slider_chng(self):
        self.lr_label.setText(f"Начальная скорость обучения: 1e{self.lr_line.value()}")

    def batch_slider_chng(self):
        self.batch_label.setText(f"Размер батча: {pow(2, self.batch_line.value())}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StartWindow()
    window.show()
    sys.exit(app.exec())
