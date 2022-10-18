a = ['a','b','c']
b = ['a']
c = []
for i in a:
    if(i not in b):
        c.append(i)

print(c)
