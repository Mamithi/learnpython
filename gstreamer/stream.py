import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import Gst, GObject

class Stream(object):
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)

        # create the element
        self.source = Gst.ElementFactory.make("rtspsrc", "source")
        self.rtph264depay_sink = Gst.ElementFactory.make("rtph264depay", "rtph264depay_sink")
        self.h264parse_sink = Gst.ElementFactory.make("h264parse", "h264parse_sink")
        self.mpegtsmux_sink = Gst.ElementFactory.make("mpegtsmux", "mpegtsmux_sink")
        self.hlssink_sink = Gst.ElementFactory.make("hlssink", "hlssink_sink")

        # Create the pipeline 
        self.pipeline = Gst.Pipeline.new("pipeline")

        # Check if the elements have been created
        if(not self.pipeline or not self.rtph264depay_sink or not self.h264parse_sink 
                or not self.mpegtsmux_sink or not self.hlssink_sink):
                print("Error: Could not create all elements")
                sys.exit(1)

        # Add the element to the pipeline (Building the pipeline)
        self.pipeline.add(self.source, self.rtph264depay_sink, self.h264parse_sink,
                 self.mpegtsmux_sink, self.hlssink_sink)
       

        # Linking the elements
        if not self.rtph264depay_sink.link(self.h264parse_sink):
            print("Error: Could not link 'rtph264depay' to 'h264parse")

        if not self.h264parse_sink.link(self.mpegtsmux_sink):
            print("Error: Could not link 'h264parse' to 'mpegtsmux")

        if not self.mpegtsmux_sink.link(self.hlssink_sink):
            print("Error: Could not link 'mpegtsmux' to 'hlssink")


        # Set properties for source and hlssink
        self.source.set_property("location", "rtsp://admin:pangani123@192.168.1.240:554/LiveMedia/ch1/Media1")
        # self.source.set_property("location", "rtsp://FACEREC:QWERTY12345@192.168.1.5:554/cam/realmonitor?channel=1&subtype=1")
        self.hlssink_sink.set_property("max-files", 5)
        self.hlssink_sink.set_property("playlist-length", 3)
        self.hlssink_sink.set_property("target-duration", 1)

        # Connect to the pad-added signal
        self.source.connect("pad-added", self.pad_added_handler)

        # start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR. Unable to set the pipeline to the playing state")
            sys.exit(1)

        # Listen to the bus
        bus = self.pipeline.get_bus()
        terminate = False
        while True:
            msg = bus.timed_pop_filtered(
                Gst.CLOCK_TIME_NONE,
                Gst.MessageType.STATE_CHANGED | Gst.MessageType.EOS | Gst.MessageType.ERROR
            )

            if not msg:
                continue

            msg_type = msg.type

            if msg_type == Gst.MessageType.ERROR:
                error, debug = msg.parse_error()
                print("ERROR: ", msg.src.get_name(), " ", error.message)
                if debug:
                    print("Debugging info:", debug)
                terminate = True
            elif msg_type == Gst.MessageType.EOS:
                print("End of stream reached")
                terminate = True
            elif msg_type == Gst.MessageType.STATE_CHANGED:
                if msg.src == self.pipeline:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    print("Pipeline state changed from {0:s} to {1:s}".format(
                        Gst.Element.state_get_name(old_state),
                        Gst.Element.state_get_name(new_state),
                    ))
            else:
                # Should not get here
                print("Error: Unexpected message received")
                break

            if terminate:
                break

        self.pipeline.set_state(Gst.State.NULL)

    def pad_added_handler(self, src, new_pad):
        print(
            "Received new pad '{0:s}' from '{1:s}'".format(
                new_pad.get_name(),
                src.get_name()
            )
        )

        # Check the new pad's type
        new_pad_caps = new_pad.get_current_caps()
        new_pad_struct = new_pad_caps.get_structure(0)
        new_pad_type = new_pad_struct.get_name()

        # create the sink pad
        sink_pad = self.rtph264depay_sink.get_static_pad("sink")

        # Check if the rtph264depay is already linked
        if(sink_pad.is_linked()):
            print("We are already linked. Ignoring....")
            return
        # Attempt the link
        ret = new_pad.link(sink_pad)
        if not ret == Gst.PadLinkReturn.OK:
            print("Type is '{0:s}' but link failed".format(new_pad_type))
        else:
            print("Link succeeded (type '{0:s}')".format(new_pad_type))

        return

if __name__ == '__main__':
    stream = Stream()

    # http://139.59.210.66/hls/spyvan1.m3u8