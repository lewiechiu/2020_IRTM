#!/usr/bin/env python
# coding: utf-8
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import NavigableString
import json
import time
import datetime
class DcardCrawler():
    def __init__(self):
        # 欲抓取的看板首頁
        self.base_url = "https://www.dcard.tw/service/api/v2/forums/mood/posts?popular=false&limit=100"
        # 抓取首頁
        self.last_post_id = ""
        self._get_last_post_id()
        
    def _get_last_post_id(self):
        
        r = requests.get(self.base_url).json()
        r = self._get_page(r)
        with open('dumped_data/{}_chunk_{}.json'.format(now.strftime("%Y-%m-%d"), 0), 'w') as outfile:
            json.dump(r, outfile, ensure_ascii=False, indent=4)
        first_post_id = r[0]['id']
        last_post_id =  r[-1]['id']
        
        for i in range(1, 1000):
            print(i, first_post_id, last_post_id)
            time.sleep(1)
            crawl_url = self.base_url + "&before={}".format(last_post_id) 
            r = requests.get(crawl_url).json()
            r = self._get_page(r)
            


            with open('dumped_data/mood_{}_chunk_{}.json'.format(now.strftime("%Y-%m-%d"), i), 'w') as outfile:
                json.dump(r, outfile, ensure_ascii=False, indent=4)
            last_post_id =  r[-1]['id']

    def _get_page(self, r):
        for page in r:
            page_url = "https://www.dcard.tw/f/mood/p/"
            web_page = requests.get(page_url + str(page['id']))
            
            soup = BeautifulSoup(web_page.text, "html.parser")
            text_element = soup.select('article> div > div > div > span ')
            if len(text_element)==0:
                continue
            else:
                text = text_element[0].text
            
            page['content'] = text

            print("##### crawling:", page['id'], "length:", len(text), "######")
            
            time.sleep(0.1)
        return r

now = datetime.datetime.now()
# print ("Current date and time : ")
DcardCrawler()



