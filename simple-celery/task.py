import time
from celery import Celery

app = Celery("task", broker='redis://localhost:6379')

@app.task
def sleep_asynchronously():
    time.sleep(20)


print("Let's begin!")

sleep_asynchronously.delay()

print(".. and thats the end of the wait")