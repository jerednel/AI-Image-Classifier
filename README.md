# AI Clothing Classifier

This is my project for the Deep Learning and AI program at DePaul.

It trains a neural network based on the Fashion MNist dataset to build a model for classifying clothing into predetermined categories.

There are 10 different options for responses.  This is a WIP.  It works with grey-scaled images that I resize manually ahead of time.  Using the cv2.resize method so I can allow for random images of color and different sizes reduces the accuracy so that is something I am exploring.

    print("I think this is a T-shirt or top")
    print("I think this is a pair of pants or jeans")
    print("I think this is a sweater/pullover")
    print("I think this is a dress")
    print("I think this is a coat")
    print("I think this is a pair of sandals...or odd looking shoe.")
    print("I think this is a shirt")
    print("I think this is a sneaker.")
    print("I think this is a bag.")
    print("I think this is an ankle boot")
