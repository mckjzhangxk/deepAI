from PyQt5 import QtWidgets

class MYWin(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        container = QtWidgets.QVBoxLayout()
        self.setLayout(container)
        for i in range(3):
            l = QtWidgets.QLabel()
            l.setText('xxxx')
            container.addWidget(l)
        self.show()


app=QtWidgets.QApplication([])
window=MYWin()
app.exec()

