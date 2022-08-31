import requests
from urllib import parse
from urllib import request  # 对照片进行下载需要的库
import os  # 对文件的操作库

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36',
    }  # 首先我们找到headers


def explanin_data(data):  # 解码函数
    image_urls = []
    for x in range(1, 9):  # 因为一张图对应八张分辨率不同的图片且都需要解码
        image_url = parse.unquote(data['sProdImgNo_%d' % x]).replace('200', '0')
        # 这里将url中的x由200替换成0 表示获取到高清壁纸
        image_urls.append(image_url)
    return image_urls


def main():
    for page in range(9):  # 我们只爬取9页，可以自由调节
        page_url = f'https://apps.game.qq.com/cgi-bin/ams/module/ishow/V1.0/query/workList_inc.cgi?activityId=2735&sVerifyCode=ABCD&sDataType=JSON&iListNum=20&totalpage=0&page={page}&iOrder=0&iSortNumClose=1&iAMSActivityId=51991&_everyRead=true&iTypeId=2&iFlowId=267733&iActId=2735&iModuleId=2735&_=1649850019827'
        resp = requests.get(page_url, headers=headers)
        # 访问网页源码
        result = resp.json()
        # json一下，变成json格式
        datas = result['List']
        # 找到result中的List标签，照片信息全部在这里
        for data in datas:
            image_urls = explanin_data(data)
            # 对照片解码并替换成高清
            name = parse.unquote(data['sProdName'])
            # 对文件名字进行解码
            dirpath = "./image"
            # os文件操作，文件明的拼接
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            # 创建文件
            for index, image_url in enumerate(image_urls):
                # 重要的函数enumerate！！！！ 可同时提取索引和内容
                request.urlretrieve(image_url, os.path.join(dirpath, "%s.jpg" % (name)))  # 进行下载
                print('%s下载完成' % image_url)


if __name__ == '__main__':
    main()
