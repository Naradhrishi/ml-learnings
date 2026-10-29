# This is a simple linear regression code where it has only one independent variable
import math


# let's define and assign a random weight and bias to the program which are called parameters in linear regression
# and known as slope and y-intercept instead of weight and bias
w = 4
b = 2

# learning rate
n = 0.015

# take few input and output data here and assign to a variable x and y 
# x is input and y is output

# X = hours studied, Y = exam score %
x = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 2.2, 3.8, 6.2]
y = [35, 38, 42, 45, 50, 52, 58, 62, 65, 68, 72, 75, 78, 82, 85, 88, 90, 93, 95, 44, 55, 74]

# now calculate the predicted value for each input and check without o/p and change accordingly
for l in range (1, 3000):

    squared_error_sum = 0
    for i in range(len(x)):
        # y-hat is the predicted value
        y_hat = x[i] * w + b
        # calculate square error here and add it to squared_error_sum variable and later use this to make average
        squared_error_sum += math.pow(y[i] - y_hat, 2)
        

    # now calculate the MSE here
    mse = squared_error_sum / len(x)
    if l % 100 == 0 or l == 1:
        print(f'Iter {l} | MSE : {mse:.4f} | w: {w:.4f} | b: {b:.4f}')


    y_hat_minus_y = []
    for j in range(len(x)):
        y_hat_minus_y.append((w * x[j] + b) - y[j])


    y_hat_minus_y_sum = 0
    for k in range(len(y_hat_minus_y)):
        y_hat_minus_y_sum += y_hat_minus_y[k]


    x_into_y_hat_minus_y_sum = 0
    for k in range(len(y_hat_minus_y)):
        x_into_y_hat_minus_y_sum += y_hat_minus_y[k] * x[k]

    # now calculate this error w.r.t w and b who is causing how much error
    error_due_to_slope = (2/len(x)) * x_into_y_hat_minus_y_sum

    error_due_to_y_intercept = (2/len(x))*y_hat_minus_y_sum


    # updated slope and y-intercept as per formula new_w = old_w - (learning_rate * error_due_to_w) and same for y-intercept as well just replace w with b

    w = w - (n * error_due_to_slope)
    b = b - (n * error_due_to_y_intercept)


# I have checked that after 2200 iterations training is DONE 

print(f"\nTraining done! Final w: {w}, b: {b}")

# Save to a simple text file - no library needed
with open("model.txt", "w") as f:
    f.write(f"{w},{b}")

print("Model saved to model.txt")

