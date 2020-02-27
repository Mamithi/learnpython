from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot

class PunchingBag(QObject):
    punched = pyqtSignal()

    def __init__(self):
        QObject.__init__(self)

    def punch(self):
        self.punched.emit()


@pyqtSlot()
def say_punched():
    print("Bag was punched.") 

bag = PunchingBag()

bag.punched.connect(say_punched)

for i in range(10):
    bag.punch()
