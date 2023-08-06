import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap


def draw_bar_plot(categories, values, color, title, figname, path, size=12):
    plt.figure()
    ax = plt.subplot(111)
    # drop top and right spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # create barplot
    x_pos = np.arange(len(categories))
    plt.bar(x_pos, values, zorder=4, color=color)

    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), color='dimgray', fontweight='bold', ha='center')

    # Create names on the x-axis
    plt.xticks(x_pos, categories, size=size, wrap=True)

    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], size=12)
    plt.ylim(0, 5)

    plt.title(title, size=12, y=1.05)
    plt.savefig(f"{path}/{figname}")


def draw_pillar_scores(results, pillar_scores, pillar_colors, path):
    for n, (pillar, sub_scores) in enumerate(results.items()):
        title = f"{pillar.capitalize()} Score {pillar_scores[pillar]}/5"
        categories = list(map(lambda x: x.replace("_", ' '), sub_scores.keys()))
        categories = ['\n'.join(wrap(l, 12, break_long_words=False)) for l in categories]
        values = list(sub_scores.values())
        nan_val = np.isnan(values)
        values = np.array(values)[~nan_val]
        categories = np.array(categories)[~nan_val]
        draw_bar_plot(categories, values, pillar_colors[n], title, f"{pillar}.png", path, size=8)
