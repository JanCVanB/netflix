"""Generate all_average.dta submission file with every rating as the average all.dta rating

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import os


def run():
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    all_data_file_path = os.path.join(root_dir, 'data/mu/all.dta')
    rating_count = 0
    rating_sum = 0
    with open(all_data_file_path, 'r') as all_data_file:
        for line in all_data_file:
            rating_count += 1
            if not (rating_count % 1000000):
                print(rating_count)
            rating_sum += int(line.split()[3])
    rating_average = rating_sum / rating_count
    ratings = ['{:.3f}\n'.format(rating_average)] * 2749898
    with open('all_average.dta', 'w+') as all_threes_submission_file:
        all_threes_submission_file.writelines(ratings)


if __name__ == '__main__':
    run()
