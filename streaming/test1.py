#!/usr/bin/env python3
# countasync.py

import asyncio

class Stream:
    def __init__(self):
        self.source = 'rtsp://admin:pangani123@192.168.1.240/LiveMedia/ch1/Media1'
        self.destination = 'rtmp://192.168.176.1/live/10' 

    async def start(self):
        stream_cmd = "ffmpeg -y -i {} -c:v copy -fflags flush_packets -fflags nobuffer -flags low_delay -tune zerolatency -an -f flv {}".format(self.source, self.destination)
        self.proc = await asyncio.create_subprocess_exec(stream_cmd)
        return self.proc



stream = Stream()
stream.start()   
