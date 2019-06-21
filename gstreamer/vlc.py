import rtsp

client = rtsp.Client('rtsp://admin:pangani123@192.168.1.240:554/LiveMedia/ch1/Media1')
try:
    client.open()
except:
    print("The source is not live")