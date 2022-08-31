import requests

# -------------------------- 更换ip --------------------------------
url = 'http://httpbin.org/ip'  # 查看ip的网址
proxy = {'http': 'http://120.220.220.95:8085'}  # 注意注意注意
resp = requests.get(url)
print(resp.text)
resp1 = requests.get(url, proxies=proxy)  # 这里我们进行参数设置
print(resp1.text)

# ----------------------- 处理不信任的SSL ----------------------------
url = 'http://inv-veri.chinatax.gov.cn'
resp = requests.get(url, verify=False)  # 只需要加上这个参数
print(resp.content.decode('utf-8'))

# ------------------------ 处理cookie -------------------------------
resp = requests.get('https://www.baidu.com/')
print(resp.cookies)
print(resp.cookies.get_dict())  # 获得cookie
