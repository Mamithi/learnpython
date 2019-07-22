import numpy as np 
logits = [2.0, 1.0, 0.1]

exps = [np.exp(i) for i in logits]

sum_of_exps = sum(exps)

softmax = [j / sum_of_exps for j in exps]

print(softmax)