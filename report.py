import os
import glob
import pandas as pd
from matplotlib import pyplot as plt


def main():
    files = glob.glob(os.path.join("images", "*.csv"))
    for file in files:
        filename = file.split(os.sep)[-1]
        df = pd.read_csv(file)
        print(filename, 'corners', len(df.index))
        plt.bar(df['x'], df['r'])
        plt.savefig(os.path.join("output", "charts", f"{filename}"))


if __name__ == "__main__":
    main()

    
