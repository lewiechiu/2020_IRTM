# coding=utf-8
import jieba
import collections
import math
import matplotlib.pyplot as plt
import jieba.analyse
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from crawlerPtt import crawlerPtt
from collections import Counter

"""目前爬蟲頁數都只有爬一頁，若要增加文本量可以將爬蟲頁數改為 5-10 頁"""

# matplotlib 中文字顯示問題
plt.rcParams['font.sans-serif']=['SimHei'] #顯示中文標籤
plt.rcParams['axes.unicode_minus']=False #顯示正負號


"""爬蟲"""
# # 輸入看板名稱及要爬蟲的頁數
board = "Movie"
url = 'https://www.ptt.cc/bbs/{}/index.html'.format(board)
crawler = crawlerPtt(url, 1)
result = crawler.getResult()

# 爬蟲結果存成檔案
file_name = "../crawlerData/ptt_{}_".format(board)+date.strftime(date.today(), format="%Y%m%d")+".pkl"
result.to_pickle(file_name)
print(f'{file_name} saved!')


"""讀取爬蟲結果"""
# 如果檔案內有一些編碼錯誤，使用 errors='ignore' 來忽略錯誤
board = "Movie"
file_name = "../crawlerData/ptt_{}_".format(board)+date.strftime(date.today(), format="%Y%m%d")+".pkl"
ptt_data = pd.read_pickle(file_name)
text = '/n'.join(ptt_data['title'])


"""斷詞"""
jieba.load_userdict('../file/userDict.txt')  #添加自訂義的詞彙
raw_words_list = jieba.cut(text)
raw_words_list = list(raw_words_list)
print('原始斷詞結果：', raw_words_list[:300])


"""字詞前處理"""
#1. 移除前後空格
# all_words_list = [word.strip() for word in raw_words_list]
all_words_list = []
for word in raw_words_list:
    clean_word = word.strip()
    all_words_list.append(clean_word)

#2. 移除空字串、標點符號、非中文字
all_words_list_no_punc = []
for word in all_words_list:
    if word != "" and word.isalpha() and '\u4e00' <= word <= '\u9fff':
        all_words_list_no_punc.append(word)


print("字詞前處理結束後：")
print(Counter(all_words_list_no_punc).most_common(30))


"""移除停用字"""
stopWords=[]
with open('../file/stopWords.txt', 'r', encoding='UTF-8') as file: #source: https://blog.droidtown.co/post/188714326387/articutnlp04
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)
all_words_list_rm_sw = []
for word in all_words_list_no_punc:
    if word not in stopWords:
        all_words_list_rm_sw.append(word)

"""關鍵字分析（方法一、觀察詞頻）"""
tf_dict = {}
for w in all_words_list_rm_sw:
    tf_dict[w] = tf_dict.get(w, 0) + 1

print("tf_dict 依照詞頻（value）排序，印出前 20 名頻繁的詞：")
tf_dict = {k: v for k, v in sorted(tf_dict.items(), key=lambda item: item[1], reverse=True)}
for key in list(tf_dict.keys())[:20]:
    print(key, tf_dict[key])


"""資料視覺化(長條圖)"""
top_keyowrds = list(tf_dict.keys())[:20]
top_keyowrds_cnt = list(tf_dict.values())[:20]
plt.figure(figsize=(10, 6))
plt.title("PTT 電影版內文關鍵字", fontsize=30)
plt.bar(top_keyowrds, top_keyowrds_cnt, width=0.8)
plt.tick_params(axis='x', direction='out', width=2, colors='black', rotation=60)
plt.tick_params(axis='both', labelsize=17)
plt.xlabel('關鍵字', fontsize=20)
plt.ylabel('總出現次數', fontsize=20)
plt.savefig("../PTT_freq.png", dpi=300)
# plt.show()


"""關鍵字分析（方法二、TF-IDF）"""
"""爬蟲爬取點影板以外的四個個看板"""
boards = ["NBA", "Stock", "creditcard", "E-shopping"]
for board in boards:
    url = 'https://www.ptt.cc/bbs/{}/index.html'.format(board)
    crawler = crawlerPtt(url, 1)
    result = crawler.getResult()
    file_name = "../crawlerData/ptt_{}_".format(board)+date.strftime(date.today(), format="%Y%m%d")+".pkl"
    result.to_pickle(file_name)
    print(f'{file_name} saved!')


"""讀取爬蟲結果"""
data = {}
boards = ["Movie", "NBA", "Stock", "creditcard", "E-shopping"]
for board in boards:
    file_name = "../crawlerData/ptt_{}_".format(board)+date.strftime(date.today(), format="%Y%m%d")+".pkl"
    ptt_data = pd.read_pickle(file_name)
    text = '/n'.join(ptt_data['content'])
    data[board] = text


"""計算計算電影版中每個字的 IDF 並存入 idf_file.txt"""
len_of_doc = len(data)
words_per_doc = {}
for board, content in data.items():
    words_in_one_doc = [w.strip() for w in jieba.cut(content)]
    words_in_one_doc = [w for w in words_in_one_doc if w != "" and w.isalpha()]
    words_per_doc[board] = set(words_in_one_doc)

# idf_dict 存 Movie 版字詞的 idf
idf_dict = {}
for w in words_per_doc['Movie']:
    idf_dict[w] = 1
    for words_in_one_doc in words_per_doc:
        if w in words_in_one_doc:
            idf_dict[w] += 1

for w, cnt in idf_dict.items():
    idf = math.log(len_of_doc / cnt)
    idf_dict[w] = idf
    
with open('../file/idf_file.txt', 'w', encoding='utf-8') as f:
    for w, idf in idf_dict.items():
        f.write(w + ' ' + str(idf) + '\n')
    print('idf_file.txt saved!')


"""利用基於 tf-idf 算法的 jieba 函式來獲取關鍵字"""
jieba.analyse.set_idf_path("../file/idf_file.txt")
keywords = jieba.analyse.extract_tags(data['Movie'], topK=100)
print('基於 tf-idf 算法的關鍵字（前 100 名）：\n', keywords)


"""資料視覺化(文字雲)"""
#背景顏色預設黑色，改為白色、使用指定字體
words = "|".join(keywords)
myWordClode = WordCloud(background_color='white',font_path='../file/SourceHanSansTW-Regular.otf').generate(words)

# 用PIL顯示文字雲
plt.figure(figsize=(10, 6))
plt.imshow(myWordClode)
plt.axis("off")
# plt.show()
# 儲存結果圖
myWordClode.to_file('../word_cloud.png')



