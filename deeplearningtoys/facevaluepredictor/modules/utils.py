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


'''打印函数'''
def Logging(message, savepath=None):
    content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
    if savepath:
        fp = open(savepath, 'a')
        fp.write(content + '\n')
        fp.close()
    print(content)


'''新建文件夹'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        return False
    return True