ss = (s for s in range(10))

aa = [a for a in range(10)]

print(aa) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(ss)

print(next(ss)) #0 (하나씩 꺼내 쓰겠다)


# 전부 꺼내 본다 
while True:
    print(next(ss))
