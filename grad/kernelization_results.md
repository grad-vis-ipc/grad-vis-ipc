| Learner             | Monks1 | Monks2 | Monks3 |
| ------------------- | ------ | ------ | ------ |
| Logistic Regression | 66     | 64     | 76     |
| KLR sq & pairs      | 89     | 63.4   | 91     |
| KLR poly(1+r/2)     | 81.7   | 66.6   | 89.8   |

| KLR R alone         | 71.7   | 65.2   | 62.6   |
| KLR sq              | 71.5   | 63.4   | 96.8   |
| KLR pairs           | 75     | 63.65  | 77.0   |
| KLR R (sq & pairs)  | 50     | 67.1   | 66.6   |


: Percent Classification Accuracy

Mine
ic| r_correlation: 0.318349
ic| r_correlation: 0.079904
ic| r_correlation: -0.08074
ic| r_correlation: -0.10715
ic| r_correlation: -0.43774
ic| r_correlation: -0.03241

sklearn
array([[1. , 0.31834887], [0.31834887, 1. ]])
array([[1. , 0.07990417], [0.07990417, 1. ]])
array([[ 1. , -0.08073974], [-0.08073974, 1. ]])
array([[ 1. , -0.1071502], [-0.1071502, 1. ]])
array([[ 1. , -0.43774017], [-0.43774017, 1. ]])
array([[ 1. , -0.03241019], [-0.03241019, 1. ]])
