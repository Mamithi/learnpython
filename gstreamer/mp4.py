import sys 
import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import Gst, GObject

class Mp4(object):
    def __init__(self):
        # Initialize gstreamer
        Gst.init(None)

        # Create the element
        self.loop = None
        self.source = Gst.ElementFactory.make("rtspsrc", "source")
        self.decoder = Gst.ElementFactory.make("rtph264depay", "decoder")
        self.parser = Gst.ElementFactory.make("h264parse", "parser")
        self.muxer = Gst.ElementFactory.make("mp4mux", "muxer")
        self.sink = Gst.ElementFactory.make("filesink", "sink")

        # Create the pipeline
        self.pipeline = Gst.Pipeline.new("Pipeline")

        # Check if all elements have been created
        if(not self.pipeline or not self.source or not self.decoder or not self.parser 
                or not self.muxer or not self.sink):
                print("Error: could not create all elements")
                sys.exit(1)

        # Build the pipeline
        self.pipeline.add(self.source, self.decoder, self.parser, self.muxer, self.sink)

        # Linking the elements
        if not self.decoder.link(self.parser):
            print("Error: could not link 'decoder' to 'parser'")
        if not self.parser.link(self.muxer):
            print("Error: could not link 'parser' to 'muxer'")
        if not self.muxer.link(self.sink):
            print("Error: could not link 'parser' to 'sink'")

        # set properties to source and filesink
        self.source.set_property("location", "rtsp://admin:pangani123@192.168.1.240:554/LiveMedia/ch1/Media1")
        self.sink.set_property("location", "file.mp4")

        # connect to the pad-added signal
        self.source.connect("pad-added", self.pad_added_handler)

        # Start playing
        self.loop = GObject.MainLoop()
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error: Unable to set the pipeline to playing state")
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
                print("Error: ", msg.src.get_name(), " ", error.message)
                if debug:
                    print("Debugging info", debug)
                self.loop.quit()
                terminate = True

            elif msg_type == Gst.MessageType.EOS:
                print("End of file reached")
                terminate = True

            elif msg_type == Gst.MessageType.STATE_CHANGED:
                if msg.src == self.pipeline:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    print("Pipeline state changed from '{0:s}' to '{1:s}'".format(
                        Gst.Element.state_get_name(old_state),
                        Gst.Element.state_get_name(new_state)
                    ))
            else:
                # should not get here
                print("Error: Unexpected message received")
                self.loop.quit()
                break

            if terminate:
                self.loop.quit()
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
        sink_pad = self.decoder.get_static_pad("sink")

        # Check if the decoder is already linked
        if(sink_pad.is_linked()):
            print("We are already linked. Ignoring...")
            return

        # Attempt the link
        ret = new_pad.link(sink_pad)
        if not ret == Gst.PadLinkReturn.OK:
            print("Type is '{0:s}' but link failed".format(new_pad_type))
        else:
            print("Link succeeded (type '{0:s}')".format(new_pad_type))

        return

if __name__ == '__main__':
    mp4 = Mp4()
