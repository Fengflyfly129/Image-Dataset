#示例，从github上下载csv文件
import pandas as pd
import io
import requests  #爬虫库
url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-10-29/nyc_squirrels.csv"
s=requests.get(url).content
print(type(s))
squirrel=pd.read_csv(io.StringIO(s.decode('utf-8')))
print(squirrel)
