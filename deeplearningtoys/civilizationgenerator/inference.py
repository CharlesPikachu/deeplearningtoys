'''
Function:
    测试文明话生成器
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import sys
import torch
import pyttsx3
import numpy as np
from modules import Poem
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui


'''文明话生成器'''
class CivilizationGenerator(QWidget):
    def __init__(self, parent=None, title='文明话生成器 —— Charles的皮卡丘', **kwargs):
        super(CivilizationGenerator, self).__init__(parent)
        rootdir = os.path.split(os.path.abspath(__file__))[0]
        self.setFixedSize(800, 500)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(os.path.join(rootdir, 'resources/icon.jpg')))
        self.poem = None
        # 定义一些必要的组件
        grid = QGridLayout()
        # --标签
        label_1 = QLabel('想要藏头的文明话:')
        label_2 = QLabel('生成的诗词最大长度:')
        # --输入框
        self.edit_1 = QLineEdit()
        self.edit_1.setText('上海牛逼全国坐牢')
        self.edit_2 = QLineEdit()
        self.edit_2.setText('132')
        # --生成按钮
        button_generate = QPushButton('生成文明话诗词')
        # --朗读按钮
        button_deacon = QPushButton('朗读生成的诗词')
        # --结果显示框
        self.text_edit = QTextEdit()
        # 组件布局
        grid.addWidget(label_1, 0, 0, 1, 1)
        grid.addWidget(self.edit_1, 0, 1, 1, 1)
        grid.addWidget(label_2, 1, 0, 1, 1)
        grid.addWidget(self.edit_2, 1, 1, 1, 1)
        grid.addWidget(button_generate, 2, 0, 1, 2)
        grid.addWidget(button_deacon, 3, 0, 1, 2)
        grid.addWidget(self.text_edit, 4, 0, 5, 2)
        self.setLayout(grid)
        # 加载数据
        tang_resource_path = os.path.join(rootdir, 'resources/tang.npz')
        poems = np.load(tang_resource_path, allow_pickle=True)
        self.dataset = poems['data']
        self.index2word = poems['index2word'].item()
        self.word2index = poems['word2index'].item()
        # 导入模型
        self.use_cuda = torch.cuda.is_available()
        self.poem_generator = Poem(len(self.word2index))
        model_path = os.path.join(rootdir, 'ckpts/best.pth')
        self.poem_generator.load_state_dict(torch.load(model_path, map_location='cpu'))
        if self.use_cuda: self.poem_generator = self.poem_generator.cuda()
        # 事件绑定
        button_generate.clicked.connect(self.generate)
        button_deacon.clicked.connect(self.deacon)
    '''朗读生成的诗词'''
    def deacon(self):
        pyttsx3.speak(self.poem)
    '''生成文明话诗词'''
    def generate(self):
        start_words, poem_list = self.edit_1.text(), []
        num_sentences = len(start_words)
        inputs, hidden = torch.LongTensor([self.word2index['<START>']]).view(1, 1), None
        if self.use_cuda: inputs = inputs.cuda()
        pre_word, pointer = '<START>', 0
        for idx in range(int(self.edit_2.text())):
            outputs, hidden = self.poem_generator(inputs, hidden)
            top_index = outputs.data[0].topk(1)[1][0].item()
            word = self.index2word[top_index]
            if pre_word in ['。', '！', '<START>']:
                if pointer == num_sentences:
                    break
                else:
                    word = start_words[pointer]
                    pointer += 1
                    inputs = (inputs.data.new([self.word2index[word]])).view(1, 1)
            else:
                inputs = (inputs.data.new([self.word2index[word]])).view(1, 1)
            poem_list.append(word)
            pre_word = word
        self.poem = ''.join(poem_list)
        self.text_edit.setText(self.poem.replace('。', '。\n'))


'''run'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = CivilizationGenerator()
    client.show()
    sys.exit(app.exec_())