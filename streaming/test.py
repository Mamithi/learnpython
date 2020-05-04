import os
import json
import subprocess
import asyncio


source = 'rtsp://admin:pangani123@192.168.1.240/LiveMedia/ch1/Media1'
destination = 'rtmp://192.168.176.1/live/10'

async def count():
    print("one")
    await asyncio.sleep(1)
    print("two")

def cli():
    errors = False
    stream_cmd = "ffmpeg -y -i {} -c:v copy -fflags flush_packets -fflags nobuffer -flags low_delay -tune zerolatency -an -f flv {}".format(source, destination)
    try: 
        # proc = subprocess.Popen([stream_cmd, '-c', 'import time; time.sleep(100)'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        proc = subprocess.Popen(stream_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, close_fds=True)


        
        # out, err = proc.communicate()
        if proc.wait() != 0:
            errors = True
            print("Error occured during streaming")
            # return err, errors
        else:
             print("Streaming started successfully...")
        # return err, errors
    except OSError as e:
        errors = True
        print("Error occured during streaming")
        # return e.strerror, errors

async def stream():
    stream_cmd = "ffmpeg -y -i {} -c:v copy -fflags flush_packets -fflags nobuffer -flags low_delay -tune zerolatency -an -f flv {}".format(source, destination)
    proc = await asyncio.create_subprocess_exec(stream_cmd)
    return proc




stream()

