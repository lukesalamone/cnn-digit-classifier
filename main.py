from run_model import RunModel



welcome =   """
            Welcome to the interactive digit classifier. We will use a convolutional neural network to classify 
            handwritten digits into one of 10 categories.
            """
runner = RunModel()

if __name__ == "__main__":
    print(welcome)
    runner.load()

    epochs = input('Number of epochs to train for (default is 2): ')
    epochs = 2 if not epochs else int(epochs)
    print('Training for %d epochs' % (epochs))

    model = runner.train(epochs)
    avg_loss, accuracy = runner.test(model, runner.testloader)
    print('accuracy: ', accuracy)