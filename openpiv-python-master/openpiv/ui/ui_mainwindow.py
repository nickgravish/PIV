# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/ui/mainwindow.ui'
#
# Created: Sun Apr 10 23:08:56 2011
#      by: PyQt4 UI code generator 4.8.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(815, 650)
        MainWindow.setMinimumSize(QtCore.QSize(815, 650))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("images/urapiv_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridlayout = QtGui.QGridLayout(self.centralwidget)
        self.gridlayout.setObjectName(_fromUtf8("gridlayout"))
        self.hboxlayout = QtGui.QHBoxLayout()
        self.hboxlayout.setObjectName(_fromUtf8("hboxlayout"))
        self.vboxlayout = QtGui.QVBoxLayout()
        self.vboxlayout.setObjectName(_fromUtf8("vboxlayout"))
        self.topCombo = QtGui.QComboBox(self.centralwidget)
        self.topCombo.setObjectName(_fromUtf8("topCombo"))
        self.vboxlayout.addWidget(self.topCombo)
        self.topGraphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.topGraphicsView.setMinimumSize(QtCore.QSize(250, 200))
        self.topGraphicsView.setObjectName(_fromUtf8("topGraphicsView"))
        self.vboxlayout.addWidget(self.topGraphicsView)
        self.bottomCombo = QtGui.QComboBox(self.centralwidget)
        self.bottomCombo.setObjectName(_fromUtf8("bottomCombo"))
        self.vboxlayout.addWidget(self.bottomCombo)
        self.statsTable = QtGui.QTableWidget(self.centralwidget)
        self.statsTable.setMinimumSize(QtCore.QSize(250, 200))
        self.statsTable.setDragDropOverwriteMode(False)
        self.statsTable.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.statsTable.setObjectName(_fromUtf8("statsTable"))
        self.statsTable.setColumnCount(0)
        self.statsTable.setRowCount(0)
        self.vboxlayout.addWidget(self.statsTable)
        self.hboxlayout.addLayout(self.vboxlayout)
        self.vboxlayout1 = QtGui.QVBoxLayout()
        self.vboxlayout1.setObjectName(_fromUtf8("vboxlayout1"))
        self.hboxlayout1 = QtGui.QHBoxLayout()
        self.hboxlayout1.setObjectName(_fromUtf8("hboxlayout1"))
        self.aFrameButton = QtGui.QPushButton(self.centralwidget)
        self.aFrameButton.setCheckable(True)
        self.aFrameButton.setChecked(False)
        self.aFrameButton.setObjectName(_fromUtf8("aFrameButton"))
        self.hboxlayout1.addWidget(self.aFrameButton)
        self.bFrameButton = QtGui.QPushButton(self.centralwidget)
        self.bFrameButton.setCheckable(True)
        self.bFrameButton.setObjectName(_fromUtf8("bFrameButton"))
        self.hboxlayout1.addWidget(self.bFrameButton)
        spacerItem = QtGui.QSpacerItem(181, 27, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.hboxlayout1.addItem(spacerItem)
        self.redrawButton = QtGui.QPushButton(self.centralwidget)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../.designer/backup")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.redrawButton.setIcon(icon1)
        self.redrawButton.setObjectName(_fromUtf8("redrawButton"))
        self.hboxlayout1.addWidget(self.redrawButton)
        self.vboxlayout1.addLayout(self.hboxlayout1)
        self.graphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.vboxlayout1.addWidget(self.graphicsView)
        self.hboxlayout2 = QtGui.QHBoxLayout()
        self.hboxlayout2.setObjectName(_fromUtf8("hboxlayout2"))
        self.startButton = QtGui.QToolButton(self.centralwidget)
        self.startButton.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../.designer/backup")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/player_start.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.startButton.setIcon(icon2)
        self.startButton.setIconSize(QtCore.QSize(32, 32))
        self.startButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.startButton.setAutoRaise(True)
        self.startButton.setObjectName(_fromUtf8("startButton"))
        self.hboxlayout2.addWidget(self.startButton)
        self.backButton = QtGui.QToolButton(self.centralwidget)
        self.backButton.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../.designer/backup")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/player_rew.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.backButton.setIcon(icon3)
        self.backButton.setIconSize(QtCore.QSize(32, 32))
        self.backButton.setAutoRaise(True)
        self.backButton.setObjectName(_fromUtf8("backButton"))
        self.hboxlayout2.addWidget(self.backButton)
        self.frameSpinBox = QtGui.QSpinBox(self.centralwidget)
        self.frameSpinBox.setMinimumSize(QtCore.QSize(80, 0))
        self.frameSpinBox.setMaximumSize(QtCore.QSize(100, 16777215))
        self.frameSpinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.frameSpinBox.setMaximum(99999)
        self.frameSpinBox.setObjectName(_fromUtf8("frameSpinBox"))
        self.hboxlayout2.addWidget(self.frameSpinBox)
        self.forwardButton = QtGui.QToolButton(self.centralwidget)
        self.forwardButton.setText(_fromUtf8(""))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../.designer/backup")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/player_fwd.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.forwardButton.setIcon(icon4)
        self.forwardButton.setIconSize(QtCore.QSize(32, 32))
        self.forwardButton.setAutoRaise(True)
        self.forwardButton.setObjectName(_fromUtf8("forwardButton"))
        self.hboxlayout2.addWidget(self.forwardButton)
        self.endButton = QtGui.QToolButton(self.centralwidget)
        self.endButton.setAutoFillBackground(False)
        self.endButton.setText(_fromUtf8(""))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../.designer/backup")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon5.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/player_end.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.endButton.setIcon(icon5)
        self.endButton.setIconSize(QtCore.QSize(32, 32))
        self.endButton.setAutoRaise(True)
        self.endButton.setObjectName(_fromUtf8("endButton"))
        self.hboxlayout2.addWidget(self.endButton)
        self.vboxlayout1.addLayout(self.hboxlayout2)
        self.hboxlayout.addLayout(self.vboxlayout1)
        self.gridlayout.addLayout(self.hboxlayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 815, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuNew = QtGui.QMenu(self.menubar)
        self.menuNew.setObjectName(_fromUtf8("menuNew"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setObjectName(_fromUtf8("menuView"))
        self.menuTools = QtGui.QMenu(self.menubar)
        self.menuTools.setObjectName(_fromUtf8("menuTools"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setWhatsThis(_fromUtf8(""))
        self.toolBar.setMovable(False)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionNew = QtGui.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(_fromUtf8("icons/filenew.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon6.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/filenew.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionNew.setIcon(icon6)
        self.actionNew.setIconVisibleInMenu(True)
        self.actionNew.setObjectName(_fromUtf8("actionNew"))
        self.actionOpen = QtGui.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(_fromUtf8("icons/fileopen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon7.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/fileopen.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionOpen.setIcon(icon7)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionSave = QtGui.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(_fromUtf8("icons/filesave.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon8.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/filesave.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionSave.setIcon(icon8)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionSaveAs = QtGui.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(_fromUtf8("icons/filesaveas.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSaveAs.setIcon(icon9)
        self.actionSaveAs.setObjectName(_fromUtf8("actionSaveAs"))
        self.actionPlay = QtGui.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_play.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPlay.setIcon(icon10)
        self.actionPlay.setObjectName(_fromUtf8("actionPlay"))
        self.actionFF = QtGui.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_fwd.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFF.setIcon(icon11)
        self.actionFF.setObjectName(_fromUtf8("actionFF"))
        self.actionFullFF = QtGui.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_end.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFullFF.setIcon(icon12)
        self.actionFullFF.setObjectName(_fromUtf8("actionFullFF"))
        self.actionPause = QtGui.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_pause.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPause.setIcon(icon13)
        self.actionPause.setObjectName(_fromUtf8("actionPause"))
        self.actionRW = QtGui.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_rew.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRW.setIcon(icon14)
        self.actionRW.setObjectName(_fromUtf8("actionRW"))
        self.actionFullRW = QtGui.QAction(MainWindow)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(_fromUtf8("icons/player_start.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFullRW.setIcon(icon15)
        self.actionFullRW.setObjectName(_fromUtf8("actionFullRW"))
        self.actionHelp = QtGui.QAction(MainWindow)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(_fromUtf8("icons/help.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHelp.setIcon(icon16)
        self.actionHelp.setObjectName(_fromUtf8("actionHelp"))
        self.actionAbput = QtGui.QAction(MainWindow)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/help.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbput.setIcon(icon17)
        self.actionAbput.setObjectName(_fromUtf8("actionAbput"))
        self.actionSetup = QtGui.QAction(MainWindow)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(_fromUtf8("icons/configure.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon18.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/configure.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionSetup.setIcon(icon18)
        self.actionSetup.setObjectName(_fromUtf8("actionSetup"))
        self.actionImageIn = QtGui.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(_fromUtf8("icons/import_image.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon19.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/import_image.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionImageIn.setIcon(icon19)
        self.actionImageIn.setObjectName(_fromUtf8("actionImageIn"))
        self.actionImportVector = QtGui.QAction(MainWindow)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap(_fromUtf8("icons/import_vector.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionImportVector.setIcon(icon20)
        self.actionImportVector.setObjectName(_fromUtf8("actionImportVector"))
        self.actionQuit = QtGui.QAction(MainWindow)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap(_fromUtf8("icons/exit.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionQuit.setIcon(icon21)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionZoomIn = QtGui.QAction(MainWindow)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap(_fromUtf8("icons/viewmag+.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon22.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/viewmag+.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionZoomIn.setIcon(icon22)
        self.actionZoomIn.setObjectName(_fromUtf8("actionZoomIn"))
        self.actionZoomOut = QtGui.QAction(MainWindow)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap(_fromUtf8("icons/viewmag-.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon23.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/viewmag-.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionZoomOut.setIcon(icon23)
        self.actionZoomOut.setObjectName(_fromUtf8("actionZoomOut"))
        self.actionProcessImages = QtGui.QAction(MainWindow)
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap(_fromUtf8("icons/kcmsystem.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon24.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/kcmsystem.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionProcessImages.setIcon(icon24)
        self.actionProcessImages.setObjectName(_fromUtf8("actionProcessImages"))
        self.actionFilterSet = QtGui.QAction(MainWindow)
        icon25 = QtGui.QIcon()
        icon25.addPixmap(QtGui.QPixmap(_fromUtf8("icons/filtersettings.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon25.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/filtersettings.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionFilterSet.setIcon(icon25)
        self.actionFilterSet.setObjectName(_fromUtf8("actionFilterSet"))
        self.actionFilter = QtGui.QAction(MainWindow)
        icon26 = QtGui.QIcon()
        icon26.addPixmap(QtGui.QPixmap(_fromUtf8("icons/filter.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon26.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/icons/filter.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionFilter.setIcon(icon26)
        self.actionFilter.setObjectName(_fromUtf8("actionFilter"))
        self.menuNew.addAction(self.actionNew)
        self.menuNew.addAction(self.actionOpen)
        self.menuNew.addAction(self.actionSave)
        self.menuNew.addAction(self.actionSaveAs)
        self.menuNew.addSeparator()
        self.menuNew.addAction(self.actionQuit)
        self.menuView.addAction(self.actionZoomIn)
        self.menuView.addAction(self.actionZoomOut)
        self.menuTools.addAction(self.actionImageIn)
        self.menuTools.addAction(self.actionSetup)
        self.menuTools.addAction(self.actionProcessImages)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbput)
        self.menubar.addAction(self.menuNew.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.toolBar.addAction(self.actionNew)
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionImageIn)
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSetup)
        self.toolBar.addAction(self.actionProcessImages)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionFilterSet)
        self.toolBar.addAction(self.actionFilter)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "OpenPiv", None, QtGui.QApplication.UnicodeUTF8))
        self.aFrameButton.setText(QtGui.QApplication.translate("MainWindow", "Frame A", None, QtGui.QApplication.UnicodeUTF8))
        self.bFrameButton.setText(QtGui.QApplication.translate("MainWindow", "Frame B", None, QtGui.QApplication.UnicodeUTF8))
        self.redrawButton.setText(QtGui.QApplication.translate("MainWindow", "Redraw", None, QtGui.QApplication.UnicodeUTF8))
        self.menuNew.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuView.setTitle(QtGui.QApplication.translate("MainWindow", "View", None, QtGui.QApplication.UnicodeUTF8))
        self.menuTools.setTitle(QtGui.QApplication.translate("MainWindow", "Tools", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setTitle(QtGui.QApplication.translate("MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBar.setWindowTitle(QtGui.QApplication.translate("MainWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8))
        self.actionNew.setText(QtGui.QApplication.translate("MainWindow", "New...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setText(QtGui.QApplication.translate("MainWindow", "Open...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave.setText(QtGui.QApplication.translate("MainWindow", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSaveAs.setText(QtGui.QApplication.translate("MainWindow", "Save As...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPlay.setText(QtGui.QApplication.translate("MainWindow", "Play", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFF.setText(QtGui.QApplication.translate("MainWindow", "FF", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFullFF.setText(QtGui.QApplication.translate("MainWindow", "FullFF", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPause.setText(QtGui.QApplication.translate("MainWindow", "Pause", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRW.setText(QtGui.QApplication.translate("MainWindow", "RW", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFullRW.setText(QtGui.QApplication.translate("MainWindow", "FullRW", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHelp.setText(QtGui.QApplication.translate("MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAbput.setText(QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSetup.setText(QtGui.QApplication.translate("MainWindow", "Setup...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSetup.setWhatsThis(QtGui.QApplication.translate("MainWindow", "Setup processing information", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImageIn.setText(QtGui.QApplication.translate("MainWindow", "Import Image...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImageIn.setWhatsThis(QtGui.QApplication.translate("MainWindow", "Import images", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImportVector.setText(QtGui.QApplication.translate("MainWindow", "Import Vector Maps...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImportVector.setWhatsThis(QtGui.QApplication.translate("MainWindow", "Import vector maps", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(QtGui.QApplication.translate("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionZoomIn.setText(QtGui.QApplication.translate("MainWindow", "Zoom In", None, QtGui.QApplication.UnicodeUTF8))
        self.actionZoomOut.setText(QtGui.QApplication.translate("MainWindow", "Zoom Out", None, QtGui.QApplication.UnicodeUTF8))
        self.actionProcessImages.setText(QtGui.QApplication.translate("MainWindow", "Process Images", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFilterSet.setText(QtGui.QApplication.translate("MainWindow", "Filter Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFilter.setText(QtGui.QApplication.translate("MainWindow", "Filter", None, QtGui.QApplication.UnicodeUTF8))

import ui_resources_rc