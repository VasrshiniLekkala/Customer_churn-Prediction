from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)

    # Save metrics
    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # Save confusion matrix
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    print("Accuracy:", acc)
