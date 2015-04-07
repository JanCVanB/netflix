"""Generate all_threes.dta submission file

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""


def run():
    threes = ['3.0\n'] * 2749898
    with open('all_threes.dta', 'w+') as all_threes_submission_file:
        all_threes_submission_file.writelines(threes)


if __name__ == '__main__':
    run()
