# 2016CCF-SouGou-大数据精准营销下的用户画像精准识别
在本次比赛中TNT_000团队获得了二等奖

团队方案如下

1.特征工程

TFIDF特征：对分词后的用户搜索词列表计算TFIDF矩阵，并通过卡方检验筛选出Top10万维特征


LDA特征：对分词后的用户搜索词列表以5为步长设置10~100的主题数目来训练LDA特征，通过连接得到用户向量



Word2vec特征：利用word2vec模型对分词后的用户搜索词列表训练100维的词向量，把每个用户搜索词的词向量做均值得到用户向量



Doc2vec特征：把分词后的用户搜索词列表当作一个文档，利用doc2vec模型训练100维的文档向量向量，得到用户向量



统计型特征：统计用户搜索词数量、英文搜索词数量等


2.模型

level1：使用TFIDF特征stack方式用LR、LinearSVC、MNB、BNB（Navie Bayes）模型得到result1

level2：使用result1和LDA、W2V、D2V和统计类型特征训练第二层模型，使用模型有CNN、XGBoost


文件目录
souGou

--1.src:源码文件存放文件夹


------excute.py:执行python文件构造count、pattern、lsi、lda、word2vec、doc2vec等特征并训练模型预测结果


------function.py:工程涉及函数逻辑python文件


------model.py:工程涉及模型逻辑python文件


------sougou_cnn_model.py:工程涉及CNN模型逻辑的python文件


------cnn.py 构造CNN stacking 特征


------fasttext.py 构造fasttext stacking特征


------tfidf.py 文件构造LR LinearSVC mnb bnb stacking特征


--2.data:存放工程数据文件夹


------data:存放原始文件和预处理后的文件


------word2vec_models:存放word2vec模型文件


--3.feature:存放特征文件夹


--4.result:存放预测结果文件夹


--5.model:存放模型结果文件夹


--6.importance:存放模型特征重要性文件夹


#---执行流程


1.配置好excute.py 里的config后 执行excute.py 先处理原始数据后再构造特征


2.执行cnn.py文件构造CNN stacking特征


3.执行fasttext.py文件构造fasttext stacking特征


4.执行tfidf.py文件构造LR LinearSVC mnb bnb stacking特征



5.构造count、pattern、lsi、lda、word2vec、doc2vec等特征


6.本地验证结果用cross-validation


7.在线预测结果提交
