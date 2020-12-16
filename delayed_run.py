from numerai_analyser.analyser import predictNumerai
import time

if __name__ == "__main__":

    wait_time_minutes = 60

    print('Current time: {}'.format(time.ctime()))

    print('Process to start in {} minutes.'.format(wait_time_minutes))

    time.sleep(wait_time_minutes * 60)

    print('Beginning process at {}'.format(time.ctime()))

    predictNumerai(False, splits = 10)

