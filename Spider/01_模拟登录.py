from urllib import request
from http.cookiejar import CookieJar
from urllib import parse

# 这个网址是登录界面
post_url = 'https://i.meishi.cc/login_t.php?redirect=https%3A%2F%2Fwww.meishij.net%2F%3Ffrom%3Dspace_block'

# 登录界面的headers和cookie
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}

cookiejar = CookieJar()
handle = request.HTTPCookieProcessor(cookiejar)
opener = request.build_opener(handle)

# 账号密码自己填充
post_data = parse.urlencode({'username': '*******',
                             'password': '*******'})

# 这里就是对username和password的填充
req = request.Request(post_url, data=post_data.encode('utf-8'))
opener.open(req)  # 执行
"""
使用requests简便登录操作
import requests
data = {'username': '*******',
        'password': '*******'}
resp = requests.post(post_url, headers=headers, data=data)
print(resp.text)
"""


url = 'https://i.meishi.cc/login_t.php?redirect=https%3A%2F%2Fwww.meishij.net%2F%3Ffrom%3Dspace_block'
rq = request.Request(url, headers=headers)
resp = opener.open(rq)
print(resp.read().decode('utf-8'))
