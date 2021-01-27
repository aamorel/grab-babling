obj_1 = [2, 3]
obj_2 = [3, 4]
a = []
a.append(obj_1)
b = []
b.append(a)
a[0] = obj_2
print(b)


obj_1 = [2, 3]
obj_2 = [3, 4]
a = []
a.append(obj_1)
b = []
b.append(a)
a = [obj_2]
print(b)
