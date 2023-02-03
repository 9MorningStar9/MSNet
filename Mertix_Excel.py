import numpy as np
import pandas as pd

io = r"G:\Mangrove\TestAccuracy\GMUSF.xls"
data = pd.read_excel(io, sheet_name=0, header=0)
# sheet_name=0 代表读取excle中的第一个sheet，header为定义列名为第0行
data1 = np.array(data)

y_true = data1[:, 1]  # 实测数据属性列索引号
y_pred = data1[:, 5]  # 分类数据属性列索引号

# true positive
TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))

# false positive
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))

# true negative
TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

# false negative
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
Accuracy = (TP + TN)/(TP + FP + TN + FN)
Error_rate = (FN + FP)/(TP + FP + TN +FN)
F1_score = 2*Precision*Recall/(Precision + Recall)
IoU = TP / (TP+FN+FP)
FWIoU = ((TP+FN)/(TP+FP+TN+FN)) * (TP/(TP+FP+FN))

po = (TP + TN)/(TP + FP + TN + FN)
pe = ((TP+FN)*(TP+FP) + (TN+FP)*(FN+TN))/((TP + FP + TN + FN)*(TP + FP + TN + FN))  # 60为单类样本的个数，120为总样本数量
Kappa = (po - pe)/(1-pe)
Confusion_matrix = np.array([[TP, FN], [FP, TN]])

print("精确率为:", Precision)
print("召回率为:", Recall)
print("总体精度为:", Accuracy)
print("F1分数为:", F1_score)
print("交并比为:", IoU)
print("频权交并比交并比为:", FWIoU)
print("Kappa系数为:", Kappa)
print("混淆矩阵为:", Confusion_matrix)
