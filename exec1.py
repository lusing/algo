sum = 0

for i in range(1, 99):
    a1 = i * (i+1) * (i+2)
    a2 = 1/ a1
    sum += a2
    print(sum)

print(sum)

prod = 1
for i2 in range(2,2016):
    a1 = i2 * (i2+2)
    a2 = 3 / a1
    a3 = 1 - a2
    prod *= a3
    print(prod)

print(prod)
print(1/prod)
print('====')

sum3 = 0
for i3 in range(1,21):
    a1 = i3 * (i3+2)
    a2 = 1/a1
    print(i3, i3+2)
    sum3 += a2
    print(sum3)

print(sum3)
