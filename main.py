import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import gui
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


IMG_H = 28
IMG_W = 28
CLASSES = open('classes.txt', 'r', encoding='utf-8').read().split()
N_CLASSES = len(CLASSES)
#print(N_CLASSES)


class App(QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.downloadButton.clicked.connect(self.openFilesDialog)
        self.recognizeButton.clicked.connect(self.recognize)
        self.saveButton.clicked.connect(self.saveFileDialog)

        self.files = []  # paths of pictures
        self.labels = []  # widgets for pictures
        self.pred_classes = [] # predict classes

    def openFilesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "",
                                                "Image Files (*.png *.jpg)", options=options)

        # clear horLayout
        for i in range(len(self.labels)):
           label = self.gridLayout.takeAt(0)
           label.widget().deleteLater()
        self.labels = []

        # display pictures
        row, col = 0, 0
        for i in range(len(self.files)):
            pixmap = QPixmap(self.files[i]).scaled(45, 45)
            self.labels.append(QLabel())
            self.gridLayout.addWidget(self.labels[i], row, col)
            col += 1
            if col % 14 == 0:
                row += 1
                col = 0
            self.labels[i].setPixmap(pixmap)
        self.gridLayout.setAlignment(QtCore.Qt.AlignCenter)


    def recognize(self):

        if len(self.files) == 0:
            error = QMessageBox()
            error.setIcon(QMessageBox.Warning)
            error.setWindowTitle("Error")
            error.setText("Error")
            error.setInformativeText("No files uploaded")
            error.exec_()
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((IMG_H, IMG_W), interpolation=2),
                transforms.ToTensor()])

            test_data = []
            for i in range(len(self.files)):
                test_data.append(transform(Image.open(self.files[i])).reshape(1, 1, IMG_H, IMG_W))

            test_data = torch.cat(test_data)

            model = torch.load('model.pth')

            probs = predict(model, test_data)
            y_pred = np.argmax(probs, -1)
            self.pred_classes = [CLASSES[i] for i in y_pred]

            self.textBrowser.setFontPointSize(24)
            self.textBrowser.setText(''.join(self.pred_classes))
            self.textBrowser.setAlignment(QtCore.Qt.AlignCenter)


    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        textFile, _ = QFileDialog.getSaveFileName(self, "Save File", "",
                                                  "Text Files (*.txt);;All Files (*)")
        fin = open(textFile, 'w', encoding='utf-8')
        fin.write(''.join(self.pred_classes))
        fin.close()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(1, x.shape)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        x = self.out(x)
        return x


def predict(model, inputs):
    with torch.no_grad():
        outputs = []
        model.eval()
        outputs.append(model(inputs))

    probs = F.softmax(torch.cat(outputs), dim=-1).numpy()
    return probs


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()