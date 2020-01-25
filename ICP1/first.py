c=input('enter name')
c=c[:-2]
c=c[::-1]
print(c)
s=input('enter the string')
c=s.replace('python','pythons')
#method
old='python'
n ='pythons'

def changeCar(s,old,new):
    r=''
    for x in s:
      if x == old:
                  x=n
      r+=x
    print(r)
#changeCar(s,'python','pythons')
#print(r1)
print(c)
x=int(input('enter no'))
y=int(input('enter 2nd no'))
x=x+y
print(x)
