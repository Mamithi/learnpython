import gi
import sys
import traceback
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

Gst.init(sys.argv)

def on_message(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.EOS:
        print("End of stream")
    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(err, debug)
    elif mtype == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(err, debug)
    return True

# create pipeline object
pipeline = Gst.Pipeline()

# Create Gst.Element by plugin name
src = Gst.ElementFactory.make("videotestsrc")

# Set property of an element
src.set_property("num-buffers", 50)
src.set_property("pattern", "snow")

sink = Gst.ElementFactory.make("gtksink")

# Add src, sink to pipeline
pipeline.add(src, sink)

# Link src with the sink
src.link(sink)

bus = pipeline.get_bus()

# Allow bus to emit messages to main thread
bus.add_signal_watch()

# Add handler to specifiv signal
bus.connect("message", on_message)

# start pipeline
pipeline.set_state(Gst.State.PLAYING)


# init GObject loop to handle Gstreamer Bus Events
loop = GObject.MainLoop()
try:
    loop.run()
except:
    traceback.print_exc()


print("Hello, its all good here")