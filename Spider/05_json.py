import json

books = [{"name": "三国演义", "price": "18"}, {"name": "水浒传", "price": "19"}]
# json.dumps将一个Python数据结构转换为JSON，json.dump() 和 json.load() 来编码和解码JSON数据,用于处理文件
result = json.dumps(books, ensure_ascii=False)
print(result)
print(type(result))

# 文件名字是books.json，w表示以写的形式读,最后显示为utf-8
fp = open("books.json", "w", encoding="utf-8")
# json.dump() 和 json.load() 来编码和解码JSON数据, 用于处理文件
json_str = json.dump(books, fp)
print(type(json_str))

# json.loads将一个JSON转换为Python数据结构
result_1 = json.loads(result)
print(result_1)
with open("books.json", "r", encoding='utf-8') as fp:
    json.load(fp)
    print(result)
