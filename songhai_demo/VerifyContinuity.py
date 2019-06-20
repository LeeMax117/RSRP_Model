# read true x,y value
import pandas as pd
testFilePath = '../resource/sample_submission.csv'
testDataSet = pd.read_csv(filepath_or_buffer=testFilePath)
# 去掉列名中的空格
testDataSet = testDataSet.rename(columns=lambda x: x.strip())
point_num = len(testDataSet)
print("point number is: %d" % point_num)  # 205行

Real_XYList = []
Real_XList = []
Real_YList = []
for row in range(0, point_num):
    tmp_list = []
    tmp_list.append(testDataSet.iloc[row][0])
    tmp_list.append(testDataSet.iloc[row][1])
    Real_XYList.append(tmp_list)
    Real_XList.append(testDataSet.iloc[row][0])
    Real_YList.append(testDataSet.iloc[row][1])
print("Real_XYList is:")
print(Real_XYList)

# 验证这些点是连续的，也就是说下一个点一定在上一个点的上下左右4个坐标之中
for index in range(1, point_num):
    x1 = Real_XList[index-1]
    y1 = Real_YList[index-1]
    x2 = Real_XList[index]
    y2 = Real_YList[index]
    if (x2 == x1 - 1) and (y2 == y1):
        print("Point %d: Left" % index)
    elif (x2 == x1 + 1) and (y2 == y1):
        print("Point %d: Right" % index)
    elif (x2 == x1) and (y2 == y1 + 1):
        print("Point %d: Up" % index)
    elif (x2 == x1) and (y2 == y1 - 1):
        print("Point %d: Down" % index)
    else:
        print("Fail: points are not continuous!")
        break
print("Success: points are continuous!")
