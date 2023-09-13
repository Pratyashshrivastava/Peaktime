from apscheduler.schedulers.background import BackgroundScheduler
import person_counter_test1 as p
import time
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.start()

job = scheduler.add_job(p.main, 'cron', minute='*/1', id='my_job_id')

print(job)

while True:
    time.sleep(5)

# create a python script that will run the main function of person_counter_test1.py every 1 minute
# and then it will automatically get locked and unlocked and then it will write the data in csv file.
# master node and slave node
# master node will run the python script and slave node will run the person_counter_test1.py

# how to run a python script from a python script ??