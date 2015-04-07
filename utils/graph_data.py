"""Graph properties and patterns of the raw data

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import matplotlib.pyplot as plt


def graph_ratings():
    num_points = 1e5
    ratings = rating_counts('data/mu/all.dta', num_points)
    rating_numbers = sorted(ratings.keys())
    x = [i - 0.4 for i in rating_numbers]
    y = [ratings[i] for i in rating_numbers]
    plt.bar(x, y, width=0.8)
    plt.title('Number of Ratings by Rating ({:n} points)'.format(num_points))
    plt.xlabel('Rating')
    plt.xlim(-0.4, 5.4)
    plt.ylabel('Number of Ratings')
    plt.show()


def rating_counts(data_file_name, num_points=float('inf'), rating_column=3):
    ratings = {}
    count = 0
    with open(data_file_name, 'r') as data_file:
        for line in data_file:
            count += 1
            if count > num_points:
                break
            values = line.split()
            rating = int(values[rating_column])
            try:
                ratings[rating] += 1
            except KeyError:
                ratings[rating] = 1
    return ratings


def run():
    graph_ratings()


if __name__ == '__main__':
    run()
