import requests
from lxml import etree
from urllib import parse
import os
from urllib import request

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
}


def main():
    """
    爬取网页中的图片
    依赖：
        pip install lxml
        pip install request
    """
    for page in range(1, 5):
        # 爬取网页的地址
        page_url = f'https://www.tupianzj.com/meinv/xinggan/list_176_{page}.html'
        # 向服务器请求数据，服务器返回的结果是个Response对象
        resp = requests.get(page_url, headers=headers)
        # 解码。因为爬取到的名字是乱码，所以在这里进行解码
        resp.encoding = resp.apparent_encoding
        # 使用xpath进行数据爬取
        text = resp.text
        parser = etree.HTML(text)
        detail_urls = parser.xpath("//div[@class='list_con_box']/ul[@class='list_con_box_ul']//li/a/img/@src")
        name = parser.xpath("//div[@class='list_con_box']/ul[@class='list_con_box_ul']/li/a/label/text()")
        for index, detail_url in enumerate(detail_urls):
            dirpath = './image'
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            # 下载图片
            request.urlretrieve(detail_url, os.path.join(dirpath, "%s.jpg" % (name[index])))
            print('%s下载完成' % detail_url)


if __name__ == '__main__':
    main()
