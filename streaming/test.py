import os
import json


with open('camera.json') as json_file:
    data = json.load(json_file)
    for camera in data["cameras"]:
        rtmp_server = data['rtmp_server'] + str(camera['id'])
        os.system("ffmpeg -y -i " + camera['rtsp_url'] + " -c:v copy -fflags flush_packets -fflags nobuffer -flags low_delay -tune zerolatency -an -f flv " + rtmp_server)




# source = 'rtsp://admin:pangani123@192.168.1.240/LiveMedia/ch1/Media1'
# destination = 'rtmp://192.168.0.39/live/1'
