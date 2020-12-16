from numerai_analyser.data_manager import NumeraiDataManager, get_latest_downloaded_comp
from numerai_analyser.config import NumeraiConfig
from numerai_analyser.analyser import predictNumerai

import pandas as pd
from datetime import datetime
import time


def compose_update(start_time, current_time, attempts, round):
    body = """Waiting for a Numerai round {0} dataset.
    - Total new round checks:\t\t{1}
    - First check performed:\t\t{2}
    - Current time:\t\t\t{3}
""".format(round, attempts, start_time, current_time)

    header = "Numerai Round {0}: still waiting for data".format(round)

    return {
        'body': body,
        'header': header,
        'html': None,
        'attachment': None
    }


if __name__ == "__main__":
    conf = NumeraiConfig(False)
    dl = NumeraiDataManager(conf)

    current_round = dl.api_conn.get_current_round()
    new_round = current_round + 1
    attempts = 0
    start_time = str(datetime.now())

    # Wait for new round data
    while not dl.api_conn.check_new_round(24):

        attempts += 1
        current_time = str(datetime.now())

        conf.send_email(**compose_update(start_time, current_time, attempts, new_round))

        # Check every thirty minutes
        time.sleep(60 * 30)

    # Perform numerai predictions on new data
    predictNumerai(False, splits = 10)

    # Wait 10 minutes for submission upload on numerai side
    time.sleep(10 * 30)

    # Check submission was successful
    current_round = dl.api_conn.get_current_round()

    submissions = pd.DataFrame(dl.api_conn.get_submission_filenames())
    round_submission = submissions[submissions.round_num == new_round]

    correct_round = new_round == current_round
    data_downloaded = current_round == get_latest_downloaded_comp('datasets')
    num_accept = round_submission.shape[0] > 0

    if correct_round and data_downloaded and num_accept:
        conf.send_email(header= 'Numerai Round {0}: Submission success'.format(current_round),
                        body = 'Predictions for round {0} have been accepted as of {1}. Congratulations!'.format(current_round,
                                                                                                                 str(datetime.now())))

    else:
        conf.send_email(header='Numerai Round {}: Submission failed'.format(current_round),
                        body="""
    Predictions for round {0} have not been accepted as of {1}. Re-running on 5 splits.
    Status:
    - Correct round number: {2}
    - Data has been downloaded: {3}
    - Numerai has accepted submission {4}""".format(current_round,
                                                    str(datetime.now()),
                                                    correct_round,
                                                    data_downloaded,
                                                    num_accept))

        predictNumerai(False, splits= 5)



