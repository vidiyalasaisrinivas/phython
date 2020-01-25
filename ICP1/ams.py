a = int(input('enter ams no'))
temp = a
sum=0
while temp > 0:
    d = temp % 10
    sum += d ** 3
    temp//=10
if sum==a:
    print('ams')
else:
    print('not ams')

