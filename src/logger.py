import csv

class CSVLogger():
    def __init__(self, fieldnames, filename='log.csv'):
        pass
        self.filename = filename
        self.csv_file = open(filename, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

    def plot_performance(self, training):
        pass

    def plot_histogram(self, V_p_pi, p_params):
        pass