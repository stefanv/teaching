columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
unique_labels = list(set(labels))
colours = ['b', 'r', 'g']
indices = [np.where(labels == l)[0] for l in unique_labels]

f, axes = plt.subplots(4, 4, figsize=(10, 10))

for column0 in range(4):
    for column1 in range(4):
        if column0 == column1:
            ax = axes[column1, column0]
            ax.set_axis_off()
            ax.text(0.5, 0.5, columns[column0], horizontalalignment='center')
            continue

        for label, index, col in zip(unique_labels, indices, colours):
            ax = axes[column1, column0]
            ax.scatter(features[index, column0],
                       features[index, column1], color=col, alpha=0.4)
