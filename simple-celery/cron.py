from celery import Celery
from celery.schedules import crontab

app = Celery()

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(30.0, test.s('Hello'), expires=10)

app = Celery("cron", broker='redis://localhost:6379')
@app.task
def test(arg):
    print(arg)