class employee:
    count=0
    sumofsalary=0
    def __init__(self,name,family,salary,department):

        self.name=name
        self.family=family
        self.salary=salary
        self.department=department
        employee.sumofsalary+=salary
        employee.count+=1
    def avg(self):
        s4=employee.sumofsalary/employee.count;
        return s4


class fulltime(employee):
      pass
      def cal(self,d):
          return d
p1=fulltime('sai','vidi',23,'clinic')
print(p1.cal(3))
#print(p1.avg())
p2=employee('sri','vidi',24,'clinic2')
print(p2.avg())