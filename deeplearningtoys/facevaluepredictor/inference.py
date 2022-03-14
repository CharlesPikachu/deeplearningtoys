'''
Function:
    模型测试
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import io
import os
import sys
import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from PIL import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui
from skimage.transform import resize


'''模型测试'''
class Inferencer(QWidget):
    def __init__(self, parent=None, title='颜值预测器 —— Charles的皮卡丘', **kwargs):
        super(Inferencer, self).__init__(parent)
        # 初始化
        self.rootdir = os.path.split(os.path.abspath(__file__))[0]
        self.setFixedSize(600, 500)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(os.path.join(self.rootdir, 'images/icon.jpg')))
        # 定义一些必要的组件
        grid = QGridLayout()
        # --图片显示用的label
        self.show_label = QLabel()
        self.show_label.setScaledContents(True)
        self.show_label.setMaximumSize(600, 400)
        self.show_image = Image.open(os.path.join(self.rootdir, 'images/assert.jpg')).convert('RGB')
        self.updateimage()
        self.show_image_ext = 'jpg'
        # --图片选择
        self.image_btn = QPushButton('选择图片')
        self.image_label = QLabel('图片路径:')
        self.image_edit = QLineEdit()
        self.image_edit.setText(os.path.join(self.rootdir, 'images/assert.jpg'))
        # --图片识别
        self.classify_btn = QPushButton('颜值预测')
        self.classify_label = QLabel('预测结果:')
        self.classify_edit = QLineEdit()
        self.classify_edit.setText('预测结果会实时更新在图片上, 最低分为0分, 最高分为5分')
        # 组件布局
        grid.addWidget(self.show_label, 0, 0, 5, 7)
        grid.addWidget(self.image_label, 5, 0, 1, 1)
        grid.addWidget(self.image_edit, 5, 1, 1, 5)
        grid.addWidget(self.image_btn, 5, 6, 1, 1)
        grid.addWidget(self.classify_label, 6, 0, 1, 1)
        grid.addWidget(self.classify_edit, 6, 1, 1, 5)
        grid.addWidget(self.classify_btn, 6, 6, 1, 1)
        self.setLayout(grid)
        # 事件绑定
        self.classify_btn.clicked.connect(self.predict)
        self.image_btn.clicked.connect(self.openimage)
    '''更新界面上的图片'''
    def updateimage(self):
        if self.show_image is None: return
        fp = io.BytesIO()
        self.show_image.save(fp, 'JPEG')
        qtimage = QtGui.QImage()
        qtimage.loadFromData(fp.getvalue(), 'JPEG')
        qtimage_pixmap = QtGui.QPixmap.fromImage(qtimage)
        self.show_label.setPixmap(qtimage_pixmap)
    '''打开图片'''
    def openimage(self):
        filepath = QFileDialog.getOpenFileName(self, '请选取图片路径', self.rootdir)
        self.image_edit.setText(filepath[0])
        self.show_image = Image.open(filepath[0]).convert('RGB')
        self.updateimage()
    '''predict'''
    def predict(self):
        # 准备工作
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # 模型初始化
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        model.load_state_dict(model_zoo.load_url('https://github.com/CharlesPikachu/deeplearningtoys/releases/download/facevaluepredictor/facevaluepredictor.pth', map_location='cpu'))
        if use_cuda: model = model.cuda()
        face_detector = dlib.get_frontal_face_detector()
        # 图片预处理
        image = cv2.imread(self.image_edit.text())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 人脸检测
        rects = face_detector(image_rgb, 1)
        if len(rects) < 1: return
        for rect in rects:
            lefttop_x, lefttop_y, rightbottom_x, rightbottom_y = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(image, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 3)
            face = image_rgb[lefttop_y: rightbottom_y, lefttop_x: rightbottom_x] / 255.
            face = resize(face, (224, 224, 3), mode='reflect')
            face = np.transpose(face, (2, 0, 1))
            face = torch.from_numpy(face).float().resize_(1, 3, 224, 224)
            face = face.type(FloatTensor)
            score = round(model(face).item(), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'Value:'+str(score), (lefttop_x-5, lefttop_y-5), font, 0.5, (0, 0, 255), 1)
        # 界面上更新结果
        self.show_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.updateimage()


'''run'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = Inferencer()
    client.show()
    sys.exit(app.exec_())