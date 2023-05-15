import queue
import sys
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from operator import attrgetter
import os
import struct as st
import zorder as zo
from queue import PriorityQueue
import heapq
import matplotlib.pyplot as plt

MAX_INT = 10000000
MIN_INT = 0
DATA = "SIM"
data_file_name = "sim"
# ep
ep = 1
dim = 2  # 维度
N = 100000  # 学习模型精度
B = 512  # 块容量
min_x = 0
min_y = 0
max_x = 7000
max_y = 1000
part_num = int(math.pow(4, math.floor(math.log(N / B, 4))))  # 部分数
print("部分数：" + str(part_num))
nx = int(math.pow(2, math.floor(math.log(N / B, 4))))  # x列数
print("行列数：" + str(nx))
ny = nx  # y列数
order = math.floor(math.log(N / B, 4))  # 曲线阶数
print("阶数:" + str(order))
# 全局块id
block_id = 0
# 全局模型id
model_id = 0
block_list = []
node_num = 0
loss_sum = 0
max_error_upper = 0
max_error_lower = 0
all_point_sum = 0
read_time = 0

# x, y分别是pos, t
# 点对象
class Point:
    x = 0.0  # pos
    y = 0.0  # t
    rank_x = 0
    rank_y = 0
    cv = 0.0
    predict_cv = 0.0
    blk = 0
    predict_blk = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.x) * hash(self.y)


# 定义一个可比较对象
class CompareAble:
    def __init__(self, priority, point):
        self.priority = priority
        self.point = point

    # python3 弃用
    # def __cmp__(self, other):
    #     if self.priority < other.priority:
    #         return -1
    #     elif self.priority == other.priority:
    #         return 0
    #     else:
    #         return 1

    # 越大越先返回
    def __lt__(self, other):
        if self.priority <= other.priority:
            return False
        else:
            return True


# 范围对象
class Rang:
    def __init__(self,x_min,x_max,y_min,y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


# 树节点
class Node:
    # 初始化一个节点
    def __init__(self, model_id, child_list, rang):
        self.child_list = child_list    # 子节点列表
        self.model_id = model_id
        self.rang = rang    # MBR

    # 添加子节点
    def add_child(self, node):
        self.child_list.append(node)
        if node is not None:
            if node.rang.x_max > self.rang.x_max:
                self.rang.x_max = node.rang.x_max
            if node.rang.x_min < self.rang.x_min:
                self.rang.x_min = node.rang.x_min
            if node.rang.y_max > self.rang.y_max:
                self.rang.y_max = node.rang.y_max
            if node.rang.y_min < self.rang.y_min:
                self.rang.y_min = node.rang.y_min

# 块
class Block:
    def __init__(self, id, child_list, rang):
        self.id = id
        self.child_list = child_list
        self.rang = rang

    def add_child(self, p):
        self.child_list.append(p)
        if p.x > self.rang.x_max:
            self.rang.x_max = p.x
        if p.x < self.rang.x_min:
            self.rang.x_min = p.x
        if p.y > self.rang.y_max:
            self.rang.y_max = p.y
        if p.y < self.rang.y_min:
            self.rang.y_min = p.y


def cal_loss(arr):
    """
    计算模型的误差
    :param arr: 点数组
    :return: 点个数， 总损失， 大于误差， 小于误差
    """
    point_sum = 0
    loss_m = 0
    error_upper = 0
    error_lower = 0
    for i in range(len(arr)):
        loss = arr[i].predict_blk - arr[i].blk          # 计算损失
        loss_m += abs(loss)                   # 计算模型总损失
        if loss > 0:
            if loss > error_upper:
                error_upper = loss          # 计算最大大于误差
        if loss < 0:
            if abs(loss) > error_lower:
                error_lower = abs(loss)     # 计算最大小于误差
        point_sum += 1
    return point_sum, loss_m, error_upper, error_lower


def construct_RSMI(arr):
    """
    构建递归模型树，神经网络模型直接保存为文件
    :param arr: 点数组
    :return: 构建好的模型树
    """
    global node_num
    print("数据量："+str(len(arr)))
    global block_id
    global model_id
    # 如果小于N，则到叶子节点，训练模型
    if len(arr) <= N:
        if len(arr) == 0:
            return None
        child_list = None
        rang = Rang(MAX_INT, MIN_INT, MAX_INT, MIN_INT)
        node = Node(model_id, child_list, rang)

        # 训练叶子模型
        print("=========训练叶子模型：" + str(model_id)+"=========")
        model = train_leaf_model(arr)
        # 保存
        # 保存叶子模型
        # model.save("model_data/model"+str(model_id)+".h5")
        model_id += 1
        # 叶子节点构建完后返回上一层
        node_num+= 1
        return node
    else:
        node = Node(model_id, [], Rang(MAX_INT, MIN_INT, MAX_INT, MIN_INT))
        print("=========训练中间模型：" + str(model_id)+"=========")
        model, part_list = train_middle_model(arr)
        # 保存中间模型
        # model.save("model_data/model"+str(model_id)+".h5")
        model_id += 1
        # 对每一个块再次递归
        # 用来记录同一层模型的序号
        id = 0
        print(len(part_list))
        for part in part_list:
            print("第"+str(id)+"部分")
            id += 1
            print("数据量："+str(len(part)))
            if part:
                leaf_node = construct_RSMI(part)
            else:
                leaf_node = None
            node.add_child(leaf_node)
        node_num+=1
        return node

def find_pos(p, X, Y):
    """
    在grid_array中查找坐标
    :param p: 点
    :param X: X刻度数组
    :param Y: Y刻度数组
    :return: grid_array中的坐标
    """
    pos = [0, 0]
    for i in range(nx):
        if p.x <= X[i]:
            pos[0] = i - 1
            break
    for i in range(ny):
        if p.y <= Y[i]:
            pos[1] = i - 1
            break
    return pos


def train_middle_model(arr):
    """
    训练中间节点模型，神经网络模型直接保存为文件
    :param arr: 点数组
    :return: 模型，分类好的点数组
    """
    global DATA
    global ep
    global model_id
    # 将arr按照数据分布分割为part_num个部分
    # gird划分格子
    grid_array = np.arange(nx * ny, dtype='int64').reshape(nx, ny)
    part_list = []
    for i in range(nx * ny):
        part_list.append([])
    print(len(part_list))
    X = [min_x]  # x刻度
    Y = [min_y]  # y刻度
    arr.sort(key=attrgetter("x"))
    for i in range(1, nx):
        X.append(arr[i * int(len(arr) / nx)].x)
    X.append(max_x)
    print("X:")
    print(X)
    arr.sort(key=attrgetter("y"))
    for i in range(1, ny):
        Y.append(arr[i * int(len(arr) / ny)].y)
    Y.append(max_y)
    print("Y:")
    print(Y)
    # 给gird_array映射Z曲线值
    for i in range(nx):
        for j in range(ny):
            grid_array[i][j] = zo.z_order_mapping(dim, order, [i, j])
    # 给每个点分配blk
    for p in arr:
        pos = find_pos(p, X, Y)
        blk = grid_array[pos[0]][pos[1]]
        p.blk = blk
    # 模型训练
    # 模型学习
    cords = []
    block = []
    for p in arr:
        cords.append([float(p.x), float(p.y)])
        block.append(int(p.blk))
        # print(float(p.x), float(p.y), int(p.blk))
    # 单隐层神经网络
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape=(2,), activation='sigmoid'),
                                 tf.keras.layers.Dense(1)]
                                )
    model.summary()
    # sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
    # 训练数据
    model.compile(optimizer='adam',
                  loss='mse')
    # checkpoint
    # 保存最好的模型
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=DATA+"/models/EP"+str(ep)+"/"+str(model_id)+".h5", monitor='loss', verbose=1,
                                                    save_best_only=True, save_weights_only=True, mode='min')
    callback_list = [checkpoint]

    model.fit(cords, block, batch_size=10000, epochs=500, callbacks=callback_list)
    # 预测块号
    pre_blk = model.predict(cords)
    print(pre_blk)
    for i in range(len(arr)):
        # 将点加入预测的块
        arr[i].predict_blk = round(pre_blk[i][0])
        print(arr[i].predict_blk)
        if arr[i].predict_blk > part_num - 1:
            arr[i].predict_blk = part_num - 1
        if arr[i].predict_blk < 0:
            arr[i].predict_blk = 0
        part_list[arr[i].predict_blk].append(arr[i])
    return model, part_list
    


def train_leaf_model(arr):
    """
    训练叶子节点的模型，神经网络模型直接保存为文件
    :param arr: 点数组
    :return: 模型
    """
    global DATA
    global ep
    global model_id
    # 继续前面的块号
    global block_id
    global block_list
    # 点排序
    # 分别按x坐标和y坐标排序，计算rank_x和rank_y
    arr.sort(key=attrgetter("x"))
    for i in range(len(arr)):
        arr[i].rank_x = i + 1
    arr.sort(key=attrgetter("y"))
    for i in range(len(arr)):
        arr[i].rank_y = i + 1
    # 计算曲线值cv
    for i in range(len(arr)):
        arr[i].cv = zo.z_order_mapping(dim, order, [int(arr[i].x), int(arr[i].y)])
    # 按照cv值排序,叶子只要cv值，不需要预测
    arr.sort(key=attrgetter("cv"))

    # 每B个点划入一个块
    for i in range(len(arr)):
        arr[i].blk = block_id + i//B
    for i in range(math.ceil(len(arr)/B)):
        block = Block(block_id, [], Rang(MAX_INT, MIN_INT, MAX_INT, MIN_INT))
        block_list.append(block)
        block_id += 1
    # 模型学习
    cords = []
    blk_list = []
    for i in range(len(arr)):
        cords.append([arr[i].x, arr[i].y])
        blk_list.append(arr[i].blk)

    # 单隐层神经网络
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape=(2,), activation='sigmoid'),
                                 tf.keras.layers.Dense(1)]
                                )
    # sgd = tf.keras.optimizers.SGD(lr=0.001, decay=0, momentum=0, nesterov=True)
    # 训练数据
    model.compile(optimizer='adam',
                  loss='mse')
    # checkpoint
    # 保存最好的模型
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=DATA+"/models/EP"+str(ep)+"/"+str(model_id)+".h5", monitor='loss', verbose=1,
                                                    save_best_only=True, save_weights_only=True, mode='min')
    callback_list = [checkpoint]

    model.fit(cords, blk_list, batch_size=10000, epochs=500, callbacks=callback_list)

    # 预测块号
    pre_blk = model.predict(cords)
    for i in range(len(arr)):
        # 将点加入预测的块
        arr[i].predict_blk = round(pre_blk[i][0])
        if arr[i].predict_blk >= len(block_list):
            block_list[len(block_list) - 1].add_child(arr[i])
        else:
            block_list[arr[i].predict_blk].add_child(arr[i])
    # 返回模型
    return model


def point_predict(p, tree):
    """
    输入点在训练好的模型树中预测点的块号
    :param p: 点
    :param tree: 模型树
    :return: 块号
    """
    global DATA
    global ep
    cord = [[float(p.x), float(p.y)]]
    model_id = tree.model_id
    # 单隐层神经网络
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape=(2,), activation='sigmoid'),
                                 tf.keras.layers.Dense(1)]
                                )
    model.load_weights(DATA+"/models/EP"+str(ep)+"/"+str(model_id)+".h5")
    num = model.predict(cord)
    num = round(num[0][0])
    # 一直预测，一直到叶子模型
    while tree.child_list:
        if num > len(tree.child_list) - 1:
            num = len(tree.child_list) - 1
        if num < 0:
            num = 0
        tree = tree.child_list[num]
        # 节点的模型id
        if tree.model_id:
            model_id = tree.model_id
        model.load_weights(DATA+"/models/EP"+str(ep)+"/"+str(model_id)+".h5")
        # 预测下一个模型
        num = model.predict(cord)
        num = round(num[0][0])
    return num


def point_query(p, tree):
    """
    输入点在训练好的模型树中查询点
    :param p: 点
    :param tree: 树
    :return: 如果没找到，则返回预测块号，如果找到了，则返回具体块号
    """
    num = point_predict(p, tree)
    # 在预测位置找
    block = block_list[num]
    for point in block.child_list:
        if point == p:
            return 1, num
    begin = num - max_error_lower
    end = num + max_error_upper
    # 在误差范围内寻找
    for i in range(begin, num):
        block = block_list[i]
        for point in block.child_list:
            if point == p:
                return 1, i
    for i in range(num + 1, end):
        block = block_list[i]
        for point in block.child_list:
            if point == p:
                return 1, i
    return -1, num


def range_query(rang, tree, data_file):
    """
    在训练好的模型树中进行范围查询
    :param rang: 范围
    :param tree: 树
    :param data_file: 数据文件
    :return: 结果点列表
    """
    global read_time
    point_list = []
    point_low = Point(rang.x_min, rang.y_min)
    point_high = Point(rang.x_max, rang.y_max)
    flag1, block_id_low = point_query(point_low, tree)
    if flag1 == 1:
        begin = block_id_low
    else:
        begin = block_id_low - max_error_lower
    if begin < 0:
        begin = 0
    flag2, block_id_high = point_query(point_high, tree)
    if flag2 == 1:
        end = block_id_high
    else:
        end = block_id_high + max_error_upper
    if end > len(block_list) - 1:
        end = len(block_list) - 1
    for i in range(begin, end):
        # block = block_list[i]
        # print(i)
        block = read_block(i, data_file)
        block = block_list[i]
        read_time += 1
        for p in block.child_list:
            if rang.x_min <= p.x <= rang.x_max and rang.y_min <= p.y <= rang.y_max:
                point_list.append(p)
    return list(set(point_list))


def read_data(file_path):
    """
    读取数据集
    :param file_path: 文件路径
    :return: 点列表
    """
    point_list = []
    min_x = 1000000
    max_x = 0
    min_y = 1000000
    max_y = 0
    dirs = os.listdir(file_path)
    for file_name in dirs:
        with open(file_path + "/" + file_name) as file_object:
            for line in file_object:
                line = line.rstrip()
                line = line.split(",")
                x = float(line[3])
                y = float(line[4])
                p = Point(x, y)
                point_list.append(p)

                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    return point_list, min_x, min_y, max_x, max_y



def write_block(arr, data_file):
    """
    将所有块依次写入硬盘
    :param arr: 块列表
    :param data_file: 数据文件
    :return: 无
    """
    for i in range(len(arr)):
        data_file.seek(i * (512 * 16))
        for p in arr[i].child_list:
            b_x = st.pack("d", p.x)
            b_y = st.pack("d", p.y)
            data_file.write(b_x)
            data_file.write(b_y)
        for i in range(512 - len(arr[i].child_list)):
            data_file.write(st.pack("d", 0))
            data_file.write(st.pack("d", 0))


def cal_depth(root):
    """
    计算模型数的深度
    :param root: 根节点
    :return: 树的深度
    """
    if root is None:
        return 0
    max_depth = 0
    if root.child_list is not None:
        # print(root.child_list)
        for child in root.child_list:
            if child is None:
                depth = 0
            else:
                depth = cal_depth(child)
            if max_depth < depth:
                max_depth = depth
        return max_depth + 1
    else:
        return 1



def read_block(id, data_file):
    """
    根据块id将块从磁盘中读出
    :param id: 块id
    :param data_file: 数据文件
    :return: 点列表
    """
    point_list = []
    data_file.seek(id * 8 * 1024)
    data_str = data_file.read(8 * 1024)
    for i in range(512):
        if len(data_str[i * 16:i * 16 + 16]) != 16:
            break
        data_tuple = st.unpack("dd", data_str[i * 16:i * 16 + 16])
        if data_tuple[0] == 0 and data_tuple[1] == 0:
            break
        p = Point(float(data_tuple[0]), float(data_tuple[1]))
        point_list.append(p)
    return point_list


def get_pmf_x(gamma, point_list):
    """
    计算x轴上的pmf函数
    :param gamma: gamma值
    :param point_list: 点表
    :return: pmf函数
    """
    pmf_x_list = []
    pmf_x_list.append(Point(0, 0))
    point_list.sort(key=attrgetter("x"))
    j = 0
    for i in range(1, gamma + 1):
        count = 0
        p = point_list[j]
        x = round((max_x - min_x) / gamma * i, 2)
        while p.x <= x:
            count+=1
            j+=1
            p = point_list[j]
        pmf_x_list.append(Point(x, count/n))
    return pmf_x_list

def pmf_x(x):
    """
    计算pmf
    :param x:输入
    :return:x轴pmf值
    """
    for i in range(len(pmf_x_list)):
        if x == pmf_x_list[i].x:
            return pmf_x_list[i].y
        if x > pmf_x_list[i].x:
            p1 = pmf_x_list[i]
            p2 = pmf_x_list[i+1]
            y = (p2.y-p1.y)/(p2.x-p1.x)*x + p1.y
            return y


def get_pmf_y(gamma, point_list):
    """
    计算y轴上的pmf函数
    :param gamma: gamma值
    :param point_list: 点表
    :return: pmf函数
    """
    pmf_y_list = []
    pmf_y_list.append(Point(0, 0))
    point_list.sort(key=attrgetter("y"))
    j = 0
    for i in range(1, gamma + 1):
        count = 0
        p = point_list[j]
        y = round((max_y - min_y) / gamma * i, 2)
        while p.y <= y:
            count+=1
            j+=1
            if j == len(point_list):
                break
            p = point_list[j]
        pmf_y_list.append(Point(y, count/n))
    return pmf_y_list

def pmf_y(x):
    """
    计算pmf
    :param x:输入
    :return:y轴pmf值
    """
    for i in range(len(pmf_y_list)):
        if x == pmf_y_list[i].x:
            return pmf_y_list[i].y
        if x > pmf_y_list[i].x:
            p1 = pmf_y_list[i]
            p2 = pmf_y_list[i+1]
            y = (p2.y-p1.y)/(p2.x-p1.x)*x + p1.y
            return y

def MINDIST(p, rang):
    """
    MINDIST距离计算
    :param p: 
    :param rang: 
    :return: 
    """
    # 1.查询点p在R内或R的边界上则MINDIST=0。
    if rang.x_min <= p.x <= rang.x_max and rang.y_min <= p.y <= rang.y_max:
        return 0
    # 2.查询点p在R外，若最短距离（p到R的边）存在，则MINDIST=p到R的边的最短距离
    # 否则，MINDIST=p到R的顶点的最短距离。
    else:
        if p.x > rang.x_max:
            if p.y > rang.y_max:
                return dist(p, Point(rang.x_max, rang.y_max))
            elif p.y < rang.y_min:
                return dist(p, Point(rang.x_max, rang.y_min))
            else:
                return p.x - rang.x_max
        if p.x < rang.x_min:
            if p.y > rang.y_max:
                return dist(p, Point(rang.x_min, rang.y_max))
            elif p.y < rang.y_min:
                return dist(p, Point(rang.x_min, rang.y_min))
            else:
                return rang.x_min - p.x
        if p.y > rang.y_max:
            return p.y - rang.y_max
        if p.y < rang.y_min:
            return rang.y_min - p.y

def dist(p1, p2):
    """
    计算两个点的距离
    :param (p1, p2): 两个点
    :return: 距离
    """
    return math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2))


def knn_query(k, q, tree, data_file):
    """
    knn查询，返回距离点q最近的k个点
    :param k: 点数量
    :param q: 中心点
    :param tree: 模型树
    :param data_file: 数据文件
    :return: 距离点q最近的k个点
    """
    global read_time
    Q = PriorityQueue(k)
    id_set = set()
    delta = 0.01
    alpha_x = delta / (pmf_x(q.x + delta) - pmf_x(q.x))
    alpha_y = delta / (pmf_y(q.y + delta) - pmf_y(q.y))
    width = alpha_x * math.sqrt(k / n)
    height = alpha_y * math.sqrt(k / n)
    while True:
        point_low = Point(q.x - width/2, q.y - height/2)
        point_high = Point(q.x + width/2, q.y + height/2)
        flag1, block_id_low = point_query(point_low, tree)
        if flag1 == 1:
            begin = block_id_low
        else:
            begin = block_id_low - max_error_lower
        if begin < 0:
            begin = 0
        flag2, block_id_high = point_query(point_high, tree)
        if flag2 == 1:
            end = block_id_high
        else:
            end = block_id_high + max_error_upper
        if end > len(block_list) - 1:
            end = len(block_list) - 1
        for i in range(begin, end):
            if i not in id_set:
                id_set.add(i)
                block = block_list[i]
                if Q.qsize() < k or MINDIST(q, block.rang) < dist(q, list(Q.queue)[0].point):
                    read_block(i, data_file)
                    read_time += 1
                    for p in block.child_list:
                        if Q.qsize() < k or dist(q, p) < dist(q, list(Q.queue)[0].point):
                            comparable = CompareAble(dist(q, p), p)
                            # 如果队列满了，则需要取出Qk，然后放入
                            if Q.full():
                                Q.get()
                                Q.put(comparable)
                            else:
                                Q.put(comparable)
        if not Q.full():
            width = width * 2
            height = height * 2
        else:
            if dist(q, list(Q.queue)[0].point) > math.sqrt(width*width + height*height)/2:
                width = 2*dist(q, list(Q.queue)[0].point)
                height = 2*dist(q, list(Q.queue)[0].point)
            else:
                break
    return Q
