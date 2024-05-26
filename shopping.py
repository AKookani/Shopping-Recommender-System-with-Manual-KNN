import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ManualNearestNeighbor import KNearestNeighbors
#from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4
DEFAULT_FILE_NAME = 'shopping.csv'


def main():

    # Check command-line arguments
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = DEFAULT_FILE_NAME

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model, MyModel = train_model(X_train, y_train)
    predictions1 = model.predict(X_test)
    predictions2 = MyModel.predict(X_test)
    sensitivity1, specificity1, F1_score1 = evaluate(y_test, predictions1)
    sensitivity2, specificity2, F1_score2 = evaluate(y_test, predictions2)

    # Print results
    print(f"Corrects of sklearn KNN model: {(y_test == predictions1).sum()}")
    print(f"Incorrects of sklearn KNN model: {(y_test != predictions1).sum()}")
    print(f"Sensitivity of sklearn KNN model rate: {100 * sensitivity1:.2f}%")
    print(f"Specificity of sklearn KNN model rate: {100 * specificity1:.2f}%")
    print(f"F1 score of sklearn KNN model: {100 * F1_score1:.2f}%")
    print(f"Corrects of My KNN model: {(y_test == predictions2).sum()}")
    print(f"Incorrects of My KNN model: {(y_test != predictions2).sum()}")
    print(f"Sensitivity of My KNN model rate: {100 * sensitivity2:.2f}%")
    print(f"Specificity of My KNN model rate: {100 * specificity2:.2f}%")
    print(f"F1 score of My KNN model: {100 * F1_score2:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    MyModel = KNearestNeighbors(k=1)
    model.fit(evidence, labels)
    MyModel.fit(evidence, labels)

    return model, MyModel


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)  # True positives
    specificity = float(0)  # True negatives
    precision = float(0)

    total_positive = float(0)   #Total actual positives
    total_negative = float(0)   #Total actual negatives

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1
                
        if prediction == 1:
            precision += (label == prediction)

    sensitivity /= total_positive
    specificity /= total_negative
    precision /= sum(predictions)
    recall = sensitivity
    
    F1_score = 2 * (precision * recall) / (precision + recall)

    return sensitivity, specificity, F1_score


if __name__ == "__main__":
    main()
