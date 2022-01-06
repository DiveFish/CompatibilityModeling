from sklearn.metrics import classification_report
import time
from typing import Union


class TrainLogger():
    def __init__(self, log_file="train-results/logs.txt"):
        self.value_dict = {}
        self.metrics_str = ""

        self.log_file = log_file

    def new_value(self, category: str, value: Union[str, int, float]):
        self.value_dict[category] = value

    def add_metrics(self, true, pred, target_names):
        self.metrics_str = classification_report(
            true, pred, target_names=target_names
        )

    def log(self):
        with open(self.log_file, "a") as logf:
            logf.write(time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime()))
            logf.write("\n\n")
            for category, value in self.value_dict.items():
                logf.write("{:20}{}".format(category, value))
                logf.write("\n")

            logf.write("\n")
            logf.write(self.metrics_str)
            logf.write("\n")
            logf.write("___________________________________________________\n\n")
