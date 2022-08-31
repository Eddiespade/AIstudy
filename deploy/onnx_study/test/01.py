s = input('input:')
l = len(s)
if l > 3:
    s = s[:2] + s[-2:]
print(s)
