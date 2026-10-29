# Loading my saved model
with open("model.txt", "r") as f:
    w, b = map(float, f.read().split(","))

# test data has two edge cases 0.5 and 11 as in train data there wasn't more than 10 in x list
X_test = [1.2, 2.8, 4.2, 5.8, 7.2, 8.8, 9.8, 1.8, 3.3, 6.8, 0.5, 11]
Y_test = [37, 49, 59, 70, 79, 89, 94, 42, 52, 77, 32, 98]

print(f"Loaded model: y = {w:.4f}*x + {b:.4f}\n")
print("Hours | Real | Predicted | Error")
print("-----------------------------------")

squared_error = 0
for x, y_real in zip(X_test, Y_test):
    y_pred = w * x + b
    error = y_real - y_pred
    squared_error += error**2
    print(f"{x:5.1f} | {y_real:4d} | {y_pred:7.1f}% | {error:+5.1f}")

mse_test = squared_error / len(X_test)
print(f"\nTest MSE : {mse_test:.4f}")

# my model training MSE was 1.78, test MSE should be close to that (2-4 is good)
if mse_test < 5:
    print("Great! Model generalizes well.")

else:
    print("Model si either overfitted or underfitted.")