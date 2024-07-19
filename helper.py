import matplotlib.pyplot as plt
from IPython import display

# 实时更新图标状况
plt.ion()

def plot(scores, mean_scores):

    # clear input
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # score -> blue
    plt.plot(scores, label='Score', color='blue')

    # mean score -> orange
    plt.plot(mean_scores, label='Mean Score', color='orange')

    plt.ylim(ymin=0)

    # display the latest score and average score
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.show(block=False)

    plt.pause(.1)