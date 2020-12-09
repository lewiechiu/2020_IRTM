#!/usr/bin/env python
# coding: utf-8
import re
import requests
import jieba
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import NavigableString


class crawlerPtt():
    def __init__(self,url,pageRange):
        # 欲抓取的看板首頁
        self.url = url
        # 儲存每個標題網址
        self.urlList = []
        # 儲存每條留言
#         self.messageList = []
        # 儲存每條留言
#         self.titleList = []
        self.data = pd.DataFrame()

        # 抓取首頁
        self.get_all_href(url=self.url)

        # 往前幾頁 抓取所有標題網址
        for page in range(1, pageRange):
            # 使用 GET 方式下載普通網頁
            r = requests.get(self.url)
            # html.parser 為解析 HTML 文件的模組
            soup = BeautifulSoup(r.text, "html.parser")
            # 抓取按鈕群
            btn = soup.select('div.btn-group > a')
            # 抓取上一頁按鈕的網址
            up_page_href = btn[3]['href']
            # 在網址後加上上一頁的網址
            next_page_url = 'https://www.ptt.cc' + up_page_href
            # 爬取上一頁內容 並更新url成上一頁的網址
            self.url = next_page_url
            self.get_all_href(url=self.url)

        # 透過標題網址self.urlList抓取留言 並儲存到self.messageList最後再存下來 (邊抓邊存會漏掉很多留言)
        self.crawlerMessage()

    # 抓取標題網址
    def get_all_href(self,url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        # 抓取文章標題
        results = soup.select("div.title")
        # 尋訪每個標題 取得網址
        for item in results:
            a_item = item.select_one("a")
            # 標題名稱
            title = item.text
            if a_item:
                #所要儲存的網站網址
                url = 'https://www.ptt.cc' + a_item.get('href')
                # 儲存網址至佇列
                self.urlList.append(url)
                
    def getResult(self):
        return self.data
    

    # 從標題網址分析留言
    def crawlerMessage(self):
        
        # [抓標題]
#         for url in self.urlList:
#             try:
#                 response = requests.get(url)
#                 soup = BeautifulSoup(response.text, 'lxml')
#                 title = soup.find('span', 'div:nth-child(3) > span.article-meta-value').getText().strip()
#                 self.titleList.append(title)
#             except:
#                     continue    
            
        
#         # [抓留言]尋訪每個標題
#         for num in range(len(self.urlList)):
#             # 建立回應
#             response = requests.get(self.urlList[num])
#             # 將原始碼做整理
#             soup = BeautifulSoup(response.text, 'lxml')
#             # 使用find_all()找尋特定目標 (推文留言)
#             articles = soup.find_all('div', 'push')
            
#             for article in articles:
#                 try:
#                     # 去除掉冒號和左右的空白
#                     message = article.find('span', 'f3 push-content').getText().replace(':', '').strip()
#                     # 把每個留言存起來
#                     self.messageList.append(message)
#                     title = article.find('span', 'div:nth-child(3) > span.article-meta-value').getText().strip()
#                     self.titleList.append(title)
#                 except:
#                     continue
            all_posts = []
            for link in self.urlList:
                try:
                    response = requests.get(link).text
                    soup = BeautifulSoup(response, 'html.parser')
                except:
                    # ignore this link
                    continue

                # meta info
                metas = soup.find_all(class_ = 'article-meta-value')
                if(len(metas) == 0):
                    continue
                title = metas[-2].text
                author = metas[0].text
                time = metas[-1].text

                # content
                content = ''
                for text in soup.find(id='main-content'):
                    if isinstance(text, NavigableString):
                        content += text.strip()

                # comments(a list of list, each includes push_tag and push_content)
                comments = []
                for div in soup.find_all(class_ = 'push'):
                    push_tag, push_content = '', ''
                    try:
                        # push_tag's type is <class 'bs4.element.ResultSet'> (?
                        push_tag = div.find('span', 'push-tag').contents[0]
                        push_content = div.find('span', 'push-content').text
                        push_tag = push_tag.strip()
                        push_content = push_content[2:]
                    except:
                        pass
                    comments.append([push_tag, push_content])

                # 將上面的每一篇文章的資訊存成一個 dictionary，並把這些 dicts 存進 all_posts (list)
                all_posts.append({'title':title, 'content':content, 'author':author, 'link':link, 'postTime':time, 'comments':comments})
            self.data = pd.DataFrame(all_posts)


