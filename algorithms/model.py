class Model:
    def load(self, file_path):
        pass

    def save(self, file_path):
        with open(file_path, 'w+') as file:
            file.write('When you change this to pickle.dump, change w+ to wb+')
