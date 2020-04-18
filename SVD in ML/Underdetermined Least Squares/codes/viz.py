import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


TITLE_FONT = FontProperties(family="Arial", size=16, weight="semibold")
AXIS_FONT = FontProperties(family="Arial", size=12)

LASALLE_BLUE = "#00245D"
GOLD = "#A48D68"


def plotPerformance(y_train, y_hat_train, avgerr_train, y_test, y_hat_test,
                    avgerr_test):
    # display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5), dpi=300)

    sorted_indices = np.argsort(np.squeeze(y_train))
    num_train = len(y_train)
    ax1.scatter(list(range(num_train)), y_train[sorted_indices], color=LASALLE_BLUE, marker='.')
    ax1.scatter(list(range(num_train)), y_hat_train[sorted_indices], color=GOLD, marker='.')
    ax1.set_title("Predictions on Joan's Known Ratings (trained with 7200 users)",
                  fontproperties=TITLE_FONT)
    ax1.set_xlabel('Jokes (sorted by true rating)', fontproperties=AXIS_FONT)
    ax1.set_ylabel('Rating', fontproperties=AXIS_FONT)
    ax1.legend(['True Rating ', 'Predicted Rating'],
               loc='upper left',
               prop=AXIS_FONT)
    ax1.axis([0, num_train, -15, 10])
    print(("Average l_2 Error (train):", avgerr_train))

    sorted_indices = np.argsort(np.squeeze(y_test))
    num_test = len(y_test)
    ax2.scatter(list(range(num_test)), y_test[sorted_indices], color=LASALLE_BLUE, marker='.')
    ax2.scatter(list(range(num_test)), y_hat_test[sorted_indices], color=GOLD, marker='.')
    ax2.set_title("Predictions on Joan's Unknown Ratings",
                  fontproperties=TITLE_FONT)
    ax2.set_xlabel('Jokes (sorted by true rating)', fontproperties=AXIS_FONT)
    ax2.legend(['True Rating ', 'Predicted Rating'],
               loc='upper left',
               prop=AXIS_FONT)
    ax2.axis([0, num_test, -15, 10])
    print(("Average l_2 Error (test):", avgerr_test))

    fig.tight_layout()
    fig.savefig("sample.png")
