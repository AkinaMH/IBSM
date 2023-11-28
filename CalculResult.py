import math as m
import re


# 预处理——读取dat
def pre_processing(filename):
    with open(filename, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        list = []
        for line in data:  # 按行读取data
            aline = re.split(r'\s*[,\s|\n]*\s', line)  # 将单个数据分隔开存好
            list.append(aline)  # 将其添加在列表之中
    return list


# 预处理——删除@行
def delete_line(data):
    count = 0
    for line in data:
        if '@' in line[0]:
            count = count + 1
    newdata = data[count:]
    return newdata


def Contpfp(data):
    pos = 0
    neg = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for n in data:
        if n[0] == 'positive':
            pos = pos + 1
        else:
            neg = neg + 1
    if pos < neg:  # negative为多数类
        for m in data:
            if m[0] == 'positive' and m[1] == 'positive':
                tp = tp + 1
            if m[0] == 'positive' and m[1] == 'negative':
                fn = fn + 1
            if m[0] == 'negative' and m[1] == 'positive':
                fp = fp + 1
            if m[0] == 'negative' and m[1] == 'negative':
                tn = tn + 1
    else:  # negative为少数类
        for m in data:
            if m[0] == 'negative' and m[1] == 'negative':
                tp = tp + 1
            if m[0] == 'negative' and m[1] == 'positive':
                fn = fn + 1
            if m[0] == 'positive' and m[1] == 'negative':
                fp = fp + 1
            if m[0] == 'positive' and m[1] == 'positive':
                tn = tn + 1
    return tp, fn, fp, tn


def ResulWrite(file_name, resdata):
    with open(file_name, "w") as f:
        for n in resdata:
            cn = 0
            while cn < len(n)-1:
                f.write(str(n[cn]))
                f.write('\t')
                cn += 1
            f.write(str(n[cn]))
            f.write('\n')
    print("保存文件成功，处理结束")


def Calresul(path, path2, path1):
    i = 0
    Fiveresult = list()
    while i < 5:
        # 拼接根目录和目标文件
        # filename = '{0}/{1}{2}/result{3}s0.tst'.format(path, path2, path1, str(i))
        filename = '{0}/{1}{2}/result{3}.tst'.format(path, path2, path1, str(i))
        data = pre_processing(filename)
        data = delete_line(data)
        tp, fn, fp, tn = Contpfp(data)
        if tp == 0:
            precision = 0
            recall = 0
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = precision * recall * 2.0 / (precision + recall)
        gmean = m.sqrt(recall * (tn * 1.0 / (tn + fp)))
        auc = 0.5 * (1 + recall - (fp * 1.0 / (fp + tn)))
        result = ['%.5f' % precision, '%.5f' % recall, '%.5f' % f1, '%.5f' % gmean, '%.5f' % auc]
        Fiveresult.append(result)
        i = i + 1
    filename2 = '{0}/{1}{2}/result.txt'.format(path, path2, path1, str(i))
    ResulWrite(filename2, Fiveresult)
    print(path1, "已经计算完毕")


# 根目录
path = 'D:/KEEL/dist/AllType/C4.5/results'


# 分类器目录
pathname = 'ADASYN-I.C45-C.'
# pathname = 'Borderline_SMOTE-I.C45-C.'
# pathname = 'OSS-I.C45-C.'
# pathname = 'OSS-I.C45-C.'
# pathname = 'SMOTE_TL-I.C45-C.'
# pathname = 'SMOTE-I.C45-C.'


# 子目录(写数据集名称)
path0 = 'abalone19'
result0 = Calresul(path, pathname, path0)
path1 = 'ecoli1'
result1 = Calresul(path, pathname, path1)
path2 = 'ecoli4'
result2 = Calresul(path, pathname, path2)
path3 = 'glass1'
result3 = Calresul(path, pathname, path3)
path4 = 'iris0'
result4 = Calresul(path, pathname,  path4)
path5 = 'pimaImb'
result5 = Calresul(path, pathname, path5)
path6 = 'vehicle0'
result6 = Calresul(path, pathname, path6)
path7 = 'wisconsinImb'
result7 = Calresul(path, pathname, path7)
path8 = 'yeast4'
result8 = Calresul(path, pathname, path8)
# path9 = 'vehicle1'
# result9 = Calresul(path,pathname,path9)
# path10 = 'vehicle3'
# result10 = Calresul(path,pathname,path10)
# path11 = 'glass-0-1-2-3_vs_4-5-6'
# result11 = Calresul(path,pathname,path11)
# path12 = 'vehicle0'
# result12 = Calresul(path,pathname,path12)
# path13 = 'ecoli1'
# result13 = Calresul(path,pathname,path13)
# path14 = 'new-thyroid2'
# result14 = Calresul(path,pathname,path14)
# path15 = 'new-thyroid1'
# result15 = Calresul(path,pathname,path15)
# path16 = 'ecoli2'
# result16 = Calresul(path,pathname,path16)
# path17 = 'segment0'
# result17 = Calresul(path,pathname,path17)
# path18 = 'glass6'
# result18 = Calresul(path,pathname,path18)
# path19 = 'yeast3'
# result19 = Calresul(path,pathname,path19)
# path20 = 'ecoli3'
# result20 = Calresul(path,pathname,path20)
# path21 = 'page-blocks0'
# result21 = Calresul(path,pathname,path21)
# path22 = 'yeast-2_vs_4'
# result22 = Calresul(path,pathname,path22)
# path23 = 'yeast-0-5-6-7-9_vs_4'
# result23 = Calresul(path,pathname,path23)
# path24 = 'vowel0'
# result24 = Calresul(path,pathname,path24)
# path25 = 'glass-0-1-6_vs_2'
# result25 = Calresul(path,pathname,path25)
# path26 = 'glass2'
# result26 = Calresul(path,pathname,path26)
# path27 = 'shuttle-c0-vs-c4'
# result27 = Calresul(path,pathname,path27)
# path28 = 'yeast-1_vs_7'
# result28 = Calresul(path,pathname,path28)
# path29 = 'glass4'
# result29 = Calresul(path,pathname,path29)
# path30 = 'ecoli4'
# result30 = Calresul(path,pathname,path30)
# path31 = 'page-blocks-1-3_vs_4'
# result31 = Calresul(path,pathname,path31)
# path32 = 'abalone9-18'
# result32 = Calresul(path,pathname,path32)
# path33 = 'glass-0-1-6_vs_5'
# result33 = Calresul(path,pathname,path33)
# path34 = 'shuttle-c2-vs-c4'
# result34 = Calresul(path,pathname,path34)
# path35 = 'yeast-1-4-5-8_vs_7'
# result35 = Calresul(path,pathname,path35)
# path36 = 'glass5'
# result36 = Calresul(path,pathname,path36)
# path37 = 'yeast-2_vs_8'
# result37 = Calresul(path,pathname,path37)
# path38 = 'yeast4'
# result38 = Calresul(path,pathname,path38)
# path39 = 'yeast-1-2-8-9_vs_7'
# result39 = Calresul(path,pathname,path39)
# path40 = 'yeast5'
# result40 = Calresul(path,pathname,path40)
# path41 = 'ecoli-0-1-3-7_vs_2-6'
# result41 = Calresul(path,pathname,path41)
# path42 = 'yeast6'
# result42 = Calresul(path,pathname,path42)
# path43 = 'abalone19'
# result43 = Calresul(path,pathname,path43)
