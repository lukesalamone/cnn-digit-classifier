import mnist



class RunModel():

    def __init__(self):
        self.loaded = False
        self.trained = False

    def load():
        print('loading images...')
        self.train_images = mnist.train_images()
        self.test_images = mnist.test_images()

        print('loading labels...')
        self.train_labels = mnist.train_labels()
        self.test_labels = mnist.test_labels()

        print('loading complete')
        self.loaded = True


    def train(epochs=100):
        self.trained = True

    def test():
        pass

    def show_metrics():
        if not self.trained:
            print('model has not been trained yet!')
            return

    def show_trained_layers():
        if not self.trained:
            print('model has not been trained yet!')
            return
