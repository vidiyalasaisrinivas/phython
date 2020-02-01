input_string = input("Enter a list numbers or elements separated by cama: ")
output=[]
userList = input_string.split(",")
for num in userList:
    x=int(num)*0.45359
    output.append(x)
print(output)