output = 0.1
output1 = 0.2
output2 = 0.3
output3 = 0.1
output4 = 0.4
output5 = 0.1
consistency=0
C=[output1, output2, output3, output4, output5]
for i in C:
    if output == i:
        consistency+=1
print(consistency/5)
