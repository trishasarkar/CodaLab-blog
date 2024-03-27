from model1 import mse as mse1
from model2 import mse as mse2
print('Conclusion:')
if mse1>mse2:
    print("Model 2 is better")
elif mse1<mse2:
    print("Model 1 is better")
else:
    print("Equivalent performance")
