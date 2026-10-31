import math
# Now we will work on multiple linear regression model building means more that 1 feature

# X - study hours and sleep and Y - sleep hours
x = [[2,7], [3,6], [4,8], [5,5], [6,7], [7,6], [8,7], [3,8], [5,6], [6,8], [2,5], [9,6]]
# X[i][0]=study hours, X[i][1]=sleep
y = [34, 37, 46, 45, 54, 57, 64, 41, 47, 56, 30, 67]

# defining slopes for each features and one y intercept of the hyperplane
w1 = 3 # For study hours
w2 = 2 # For sleep hours

b = 5 # initial bias / y-intercept of the hyperplane

# define learning rate here
n = 0.01


for l in range(0, 70000):

    squared_error_sum = 0

    for i in range(len(x)):
        y_hat = x[i][0] * w1 + x[i][1] * w2 + b
        squared_error_sum += math.pow(y[i] - y_hat, 2)

    # Now calculating mean square error(MSE)
    mse = squared_error_sum / len(x)

    if l % 500 == 0 or l == 1:
        print(f'Iter {l} | MSE : {mse:.4f} | w1: {w1:.4f} | w2: {w2:.4f} | b: {b:.14f}')


    y_hat_minus_y = []
    for i in range(len(x)):
        y_hat_minus_y.append(( x[i][0] * w1 + x[i][1] * w2 + b) - y[i])

    y_hat_minus_y_sum = 0
    for k in range(len(y_hat_minus_y)):
        y_hat_minus_y_sum += y_hat_minus_y[k]

    x1_into_y_hat_minus_y_sum = 0
    for k in range(len(y_hat_minus_y)):
        x1_into_y_hat_minus_y_sum += y_hat_minus_y[k] * x[k][0]

    x2_into_y_hat_minus_y_sum = 0
    for k in range(len(y_hat_minus_y)):
        x2_into_y_hat_minus_y_sum += y_hat_minus_y[k] * x[k][1]

    error_due_to_w1 = (2/len(x)) * x1_into_y_hat_minus_y_sum
    error_due_to_w2 = (2/len(x)) * x2_into_y_hat_minus_y_sum

    error_due_to_b = (2/len(x)) * y_hat_minus_y_sum

    # Now update w1, w2 and bias - b 
    w1 = w1 - (n * error_due_to_w1)
    w2 = w2 - (n * error_due_to_w2)
    b = b - (n * error_due_to_b)


# I have checked that after near around 70000 epoch training is DONE (almost perfect 99.9999% accurate)

print(f"\nTraining done! Final w1: {w1}, w2: {w2}, b: {b}")

with open("model.txt", "w") as f:
    f.write(f"{w1},{w2},{b}")

print("Model saved to model.txt")