'''
Function:
    一些工具函数
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import time


'''新建文件夹'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        return False
    return True


'''打印函数'''
def Logging(message, savepath=None):
    content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
    if savepath:
        fp = open(savepath, 'a')
        fp.write(content + '\n')
        fp.close()
    print(content)


'''保存所有类别'''
def saveClasses(classes, filename='classes.data'):
    with open(filename, 'w') as f:
        for c in classes:
            f.write(c + '\n')
    return True


'''导入所有类别'''
def loadClasses(filename='classes.data'):
    classes = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip('\n'):
                classes.append(line.strip('\n'))
    return classes