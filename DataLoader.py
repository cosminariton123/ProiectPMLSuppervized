
def load_submission_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        file = f.read()
        file = file.split("\n")
        data = [line.split("\t")[4] for line in file]
        data = [line.split(" ") for line in data]
        idx = [line.split("\t")[0] for line in file]
    return idx, data

def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        file = f.read()
        file = file.split("\n")
        data = [line.split("\t")[4] for line in file]
        data = [line.split(" ") for line in data]
        labels = [float(line.split("\t")[-1]) for line in file]
    return data, labels

def load_training_data(filename):
    data, labels = load_data(filename)

    train_data = data[:int(len(data) * 80/100)]
    train_labels = labels[:int(len(labels) * 80/100)]

    validation_data = data[int(len(data) * 80/100):]
    validation_labels = labels[int(len(labels) * 80/100):]

    return train_data, train_labels, validation_data, validation_labels


def make_submission(idx, predicted_labels):
    with open("submission.txt", "w") as f:
        f.write("id,label\n")
        
        for id, predicted_label in zip(idx, predicted_labels):
            f.write(id)
            f.write(",")
            f.write(str(predicted_label))
            
            if id != idx[-1]:
                f.write("\n")
