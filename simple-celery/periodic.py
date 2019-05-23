from celery import Celery

app = Celery('periodic', broker="redis://localhost:6379")

@app.task
def see_you():
    print("See you in ten seconds")

app.conf.beat_schedule = {
    "see-you-in-ten-seconds-task" : {
        "task": "periodic.see_you",
        "schedule": 10.0
    }
}