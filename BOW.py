import syllables

class BOW():
    def __init__(self):
        self.vocabulary = dict()

    def extract_vocabulary(self, train_data):
        for line in train_data:
            for word in line:
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = len(self.vocabulary)
    
    def extract_features_3(self, data):
        features = list()
        for line in data:
            line_histogram_and_features = [0 for _ in self.vocabulary.keys()]
            #word histogram
            for word in line:
                if word in self.vocabulary.keys():
                    line_histogram_and_features[self.vocabulary[word]] += 1
            features.append(line_histogram_and_features)
        return features

    def extract_features_2(self, data):
        features = list()
        for line in data:
            line_histogram_and_features = [0 for _ in self.vocabulary.keys()]
            #word histogram
            for word in line:
                if word in self.vocabulary.keys():
                    line_histogram_and_features[self.vocabulary[word]] += 1
            #nr of words
            line_histogram_and_features.append(len(line))
            features.append(line_histogram_and_features)
        return features

    def extract_features(self, data):
        features = list()
        for line in data:
            line_histogram_and_features = [0 for _ in self.vocabulary.keys()]
            #word histogram
            for word in line:
                if word in self.vocabulary.keys():
                    line_histogram_and_features[self.vocabulary[word]] += 1
            #nr of words
            line_histogram_and_features.append(len(line))

            #capital letters
            capital_letters = 0
            for word in line:
                if word[0].isupper() == True:
                    capital_letters += 1
            line_histogram_and_features.append(capital_letters)

            #nr of vowels + nr of consonants
            nr_of_vowels = 0
            nr_of_consonants = 0
            for word in line:
                for letter in word:
                    if letter.upper() in ["A", "E", "I", "O", "U"]:
                        nr_of_vowels += 1
                    else:
                        nr_of_consonants += 1
            line_histogram_and_features.append(nr_of_vowels)
            line_histogram_and_features.append(nr_of_consonants)

            #nr_of_syllables
            nr_of_syllables = 0
            for word in line:
                nr_of_syllables += syllables.estimate(word)      
            line_histogram_and_features.append(nr_of_syllables)  

            features.append(line_histogram_and_features)
        return features