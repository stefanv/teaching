centroid, kmeans_labels = kmeans2(data, 3)

colors = ['m', 'y', 'c', 'r', 'g']

for n, l in enumerate(set(kmeans_labels)):
    ix = np.where(kmeans_labels == l)[0]
    plt.scatter(data[ix, 0], data[ix, 1], c=colors[n])
