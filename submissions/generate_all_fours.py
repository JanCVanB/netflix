"""Generate all_threes.dta submission file with every rating as 3.0

.. moduleauthor:: Quinn Osha stolen from Jan Van Bruggen <jancvanbruggen@gmail.com>
"""


def run():
    fours = ['4.0\n'] * 2749898
    with open('all_fours.dta', 'w+') as all_fours_submission_file:
        all_fours_submission_file.writelines(fours)


if __name__ == '__main__':
    run()
