
def sum_digits(n):
    return sum(int(d) for d in str(n))
        
smallest_number = None
for i in range(1, 1000):
    if i == sum_digits(i):        
      smallest_number = i        
      break
smallest_number = i
