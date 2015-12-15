import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

# The cost function is in the form: 
# J = class * log(hypothesis) + (1-class) * log(1 - hypothesis)
# If the class is 1 the cost will be given by log(hypothesis), so the cost will be close to 0 if the 
# hypothesis is close to 1 and close to infinite if hypothesis is too far from 1. When class is 0, 
# the cost will be log(1-hypothesis), then the cost will be low when hypothesis approaches 0.
def cost_function(training, classes, theta):
    m = len(classes)

    hypothesis = vsigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), np.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1, classes)), np.log(np.subtract(1,hypothesis)))
    J = -1/m * (j_1[0][0] + j_2[0][0])
    
    return J

# The regularized cost funtion adds a regularization term lambda to the previous cost function 
# in order to avoid overfitting
def cost_function_reg(training, classes, theta, regLambda):
    m = len(classes)

    # theta2 excludes the first parameter in orther to be regularized
    theta2 = theta[range(1,theta.size)]

    hypothesis = vsigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), np.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1,classes)), np.log(np.subtract(1,hypothesis)))
    j_3 = (regLambda/(2*m)) * np.sum(theta2**2)
    J = (1/m) * (-1) * np.add(j_1, j_2) + j_3    

    return J

# gradient_descent updates the value of theta by subtracting a gradient of the cost from the 
# previous value of theta 
def gradient_descent(training, classes, theta, alpha, num_iterations):
    m = len(classes)

    for i in range(0,num_iterations):

        hypothesis = vsigmoid(np.dot(training,theta))
        
        gradient = np.multiply(1/m, np.dot(np.transpose(np.subtract(hypothesis, classes)),training))

        theta = np.subtract(theta, np.transpose(np.multiply(alpha, gradient)))

    return theta

# Regularized gradient descent
def gradient_descent_reg(training, classes, theta, alpha, num_iterations, regLambda):
    m = len(classes)

    for i in range(0,num_iterations):

        hypothesis = vsigmoid(np.dot(training,theta))

        gradient_1 = np.multiply(1/m, np.dot(np.transpose(np.subtract(hypothesis, classes)),training))
        gradient_2 = np.multiply((regLambda/m), theta)
        gradient = np.transpose(gradient_1) + gradient_2
        gradient[0] = (1/m) * np.dot(np.transpose(np.subtract(hypothesis, classes)),np.array(training[:,0]))

        theta = np.subtract(theta, np.multiply(alpha, gradient))

    return theta

def classify(testing, theta):
    return vsigmoid(np.dot(testing,theta))


