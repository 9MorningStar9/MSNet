import os
from osgeo import gdal
import numpy as np
import time

os.environ['PROJ_LIB'] = r"D:\Anaconda3.9\envs\tensorflow_gpu\Library\share\proj"


# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    return dataset


# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:  # 多波段
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:  # 单波段
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


# 像素坐标和地理坐标仿射变换
def CoordTrans(X_Pixel, Y_Pixel, GeoTransform):
    XGeo = GeoTransform[0] + GeoTransform[1] * X_Pixel + Y_Pixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * X_Pixel + Y_Pixel * GeoTransform[5]
    return XGeo, YGeo


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

    # 获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1
    # 裁剪图片,重复率为RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            # 如果图像是单波段
            if len(img.shape) == 2:
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            # 如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            XGeo, YGeo = CoordTrans(int(j * CropSize * (1 - RepetitionRate)), int(i * CropSize * (1 - RepetitionRate)),
                                    geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
            # 写图像
            writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
            # 文件名 + 1
            new_name = new_name + 1
    # 向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if len(img.shape) == 2:
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]

        XGeo, YGeo = CoordTrans(width - CropSize, int(i * CropSize * (1 - RepetitionRate)), geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
        # 写图像
        writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1
    # 向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if len(img.shape) == 2:
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:, (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        XGeo, YGeo = CoordTrans(int(j * CropSize * (1 - RepetitionRate)), height - CropSize, geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
        writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
        # 文件名 + 1
        new_name = new_name + 1
    # 裁剪右下角
    if len(img.shape) == 2:
        cropped = img[(height - CropSize): height, (width - CropSize): width]
    else:
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]

    XGeo, YGeo = CoordTrans(width - CropSize, height - CropSize, geotrans)
    crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
    new_name = new_name + 1
    writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)


begin = time.time()
TifCrop("./Predict/origin_image_4096/1.tif", "./result-example/origin", 256, 0.0)  # 影像
TifCrop("./Predict/FCN/result/1.tif", "./result-example/FCN", 256, 0.0)  # 影像
TifCrop("./Predict/SegNet/result/1.tif", "./result-example/SegNet", 256, 0.0)  # 影像
TifCrop("./Predict/Unet/result/1.tif", "./result-example/Unet", 256, 0.0)  # 影像
TifCrop("./Predict/MRCNN256/result/1.tif", "./result-example/MRCNN256", 256, 0.0)  # 影像
# TifCrop("./Pki_Mangrove_image/research_area.tif", "./Predict/origin_image_4050", 4050, 0.1)  # 影像
# 4050, 4096(9*9预测图像大小，256*256预测图像大小)
end = time.time()
print("裁剪耗时：%d秒" % (end - begin))
