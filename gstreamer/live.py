import sys
import gi

gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import Gst, GObject


class Live(object):
    def __init__(self):
        Gst.init(None)

        self.source = Gst.ElementFactory.make("rtspsrc", "source")
        # self.source = Gst.ElementFactory.make("v4l2src", "source")
        # self.sink = Gst.ElementFactory.make("autovideosink", "sink")

        self.pipeline = Gst.Pipeline.new("pipeline")

        if not self.pipeline or not self.source:
            print("Error: Could not create the elements")

        # self.pipeline.add(self.source, self.sink)
        self.pipeline.add(self.source)
        #
        # if not self.source.link(self.sink):
        #     print("ERROR: Could not link source and sink")

        self.source.set_property("location", "rtsp://admin:pangani123@192.168.1.240:554/LiveMedia/ch1/Media1")
        # self.source.set_property("device", "/dev/video0")

        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR. Unable to set the pipeline to the playing state")
            sys.exit(1)

        

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
                    print("Debugging info", debug)
                terminate = True


            elif msg_type == Gst.MessageType.EOS:
                print("End of stream reached")
                terminate = True

            elif msg_type == Gst.MessageType.STATE_CHANGED:
                if msg.src == self.pipeline:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    print("Pipeline stated changed from {0:s} to {1:s}".format(
                        Gst.Element.state_get_name(old_state),
                        Gst.Element.state_get_name(new_state)
                    ))

            else:
                print("ERROR: unexpected message received")
                break
            if terminate:
                break
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    live = Live()
