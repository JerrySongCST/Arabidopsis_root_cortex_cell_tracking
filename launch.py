import time

import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PySide2.QtGui import QIcon
from PySide2.QtCore import Signal,QObject
from torch import cuda, device, load
from ga import *
from cell_detection import *
from time import sleep
from threading import Thread
from pyqtgraph import GraphicsLayoutWidget, mkPen, ScatterPlotItem, mkColor
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import sys
from multiprocessing.pool import Pool
import multiprocessing

multiprocessing.freeze_support()


# 信号库
class SignalStore(QObject):
    # 定义一种信号
    progress_update = Signal(int)
    # 还可以定义其他作用的信号

class TerminateSignal(QObject):
    terminate = Signal()

# 实例化
so = SignalStore()
so1 = SignalStore()
so2 = SignalStore()
t1 = TerminateSignal()
t2 = TerminateSignal()


class Tracking(QMainWindow):
    def __init__(self, directory, length):
        so1.progress_update.connect(self.setProgress)
        t1.terminate.connect(self.closeui)
        self.base_direc = directory
        self.time_intervel = float(get_settings(directory)[0].getAttribute("timeinterval"))
        self.length = length
        self.ongoing = False
        loader = QUiLoader()
        self.ui = loader.load('tracking.ui')
        self.ui.progressBar.hide()
        self.ui.progressBar.setRange(0, self.length)
        self.ui.link_lines.clicked.connect(self.link)
        self.ui.track_cells.hide()
        self.ui.groupBox.hide()
        self.ui.save.hide()
        self.ui.track_cells.clicked.connect(self.track)
        self.ui.save.clicked.connect(self.save)
        img_path = get_imlist(self.base_direc)[0]
        self.imgs = io.imread(img_path)
        self.img_shape = np.shape(self.imgs)

    def setProgress(self, value):
        self.ui.progressBar.setValue(value)

    def closeui(self):
        QMessageBox.information(
            self.ui,
            'successful',
            'Tracking is completed! Congrats!')
        self.ui.close()

    def link(self):
        self.ui.progressBar.show()
        new_ranks = link_lines(self.length, self.base_direc)
        def workerThreadFunc():
            self.ongoing = True
            for i in range(self.length):
                sleep(1)
                # 设置进度值
                line_correspondence(self.base_direc, i, new_ranks[i])
                so1.progress_update.emit(i+1)
                # self.ui.progressBar.setValue(i + 1)
            self.ongoing = False
            self.ui.track_cells.show()
            self.ui.groupBox.show()
            self.ui.link_lines.hide()
        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return
        worker = Thread(target=workerThreadFunc)
        worker.start()

    def track(self):
        self.ui.progressBar.hide()
        self.ui.progressBar.setRange(0, self.length-1)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.show()
        def workerThreadFunc():
            self.ongoing = True
            get_mitotic_color(self.base_direc)
            all_spots = track_cells(self.length, self.base_direc)
            # nethermost = nethermost_8_cells(np.array(all_spots[0]))
            nethermost, zs, xs = nethermost_8_cells_y(np.array(all_spots[0]))
            nethermost_index = [0 for _ in range(8)]

            # dev = device("cuda" if cuda.is_available() else "cpu")
            # model = UNet()
            # model.to(dev)
            # # check_point = load(r'./u_net_cell_best_new_zdouble.pth')
            # check_point = load(r'./unet_cell_best_new.pth')
            # model.load_state_dict(check_point['state_dict'])
            # model.eval()
            # pred_masks = []
            # for i in range(self.length):
            #     mask = detect_middle_slice(model, self.base_direc, i, self.imgs[i][int(self.img_shape[1]/2)], dev)
            #     pred_masks.append(mask)

            for i in range(self.length - 1):
                sleep(1)
                print(f"Current image is {i}")
                reset_excels(i, self.base_direc)
                cells_before = np.array(all_spots[i])
                cells_later = np.array(all_spots[i + 1])
                if self.ui.radioButton_2.isChecked():
                    link_spots(cells_before, cells_later, i, self.base_direc)
                else:
                    off_set = []
                    for q in range(8):
                        off_set.append(translation_calculate(self.imgs[i][zs[q]], self.imgs[i+1][zs[q]], i))
                        # off_set.append(affine_calculate(self.imgs[i][zs[q]], self.imgs[i+1][zs[q]], i, q,
                        #                                 [xs[q], nethermost[q]]))
                    # nethermost, nethermost_index = link_spots_by_xy(cells_before, cells_later, i, self.base_direc,
                    #                                                 nethermost, nethermost_index, off_set)

                    nethermost, nethermost_index, zs, xs = link_spots_by_y(cells_before, cells_later, i, self.base_direc,
                                                                           nethermost, nethermost_index, off_set)
                so1.progress_update.emit(i + 1)
                # self.ui.progressBar.setValue(i + 1)
            self.ongoing = False
            self.ui.save.show()
            self.ui.track_cells.hide()
            self.ui.groupBox.hide()
        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return
        worker = Thread(target=workerThreadFunc)
        worker.start()


    def save(self):
        self.ui.progressBar.hide()
        self.ui.progressBar.setRange(0, self.length - 1)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.show()
        filePath, _ = QFileDialog.getSaveFileName(
            self.ui,  # 父窗口对象
            "save xml file",  # 标题
            r"{0}".format(self.base_direc),  # 起始目录
            "xml type (*.xml)"  # 选择类型过滤项，过滤内容在括号中
        )

        def workerThreadFunc():
            xml = get_imlist(self.base_direc, '.xml')
            domTree = ps(xml[0])
            all_cells = []
            tracked_cells = []
            track_id = 0
            self.ongoing = True
            for i in range(self.length):
                cells = read_excels_tracked_cell(i, self.base_direc)
                all_cells.append(cells)
            all_cells[0] = rank_line_then_y(all_cells[0])
            domTree = judge_blank_track(domTree)
            # domTree.write(f"{filePath}")
            for i in range(self.length-1):
                sleep(1)
                tracked_cells, track_id, domTree = tracked_cells_to_xml(i, all_cells, tracked_cells,
                                                                        self.length, domTree, track_id,
                                                                        self.time_intervel)
                so1.progress_update.emit(i + 1)
                #self.ui.progressBar.setValue(i + 1)
            domTree.write(f"{filePath}")
            self.ongoing = False
            t1.terminate.emit()
        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return

        worker = Thread(target=workerThreadFunc, daemon=True)
        worker.start()
        # worker.join()


class Clustering(QMainWindow):
    def __init__(self, selected_direc):
        so.progress_update.connect(self.setProgress)
        self.ongoing = False
        self.colorlist = eight_colors()
        loader = QUiLoader()
        loader.registerCustomWidget(GraphicsLayoutWidget)
        loader.registerCustomWidget(GLViewWidget)
        self.ui = loader.load('clustering.ui')
        self.base_direc = selected_direc
        self.set_text()
        self.ui.toolButton.clicked.connect(self.choosefile)
        self.length = length_imgs(self.base_direc)[0]
        self.ui.progressBar.hide()
        self.ui.progressBar.setRange(0, self.length)
        self.ui.pushButton.clicked.connect(self.clustering)
        self.ui.clustering_check.clicked.connect(self.clustering_check)
        self.ui.refresh.hide()
        self.ui.previous.hide()
        self.ui.next.hide()
        self.setcolor()
        self.hide_color_buttons()
        self.current = 0
        self.showcurrent()
        self.clustered_frames = []
        self.w1 = self.ui.graphicsView.addPlot()
        self.ui.refresh.clicked.connect(self.clustering_check)
        self.ui.sort.clicked.connect(self.sort_id)
        self.ui.next.clicked.connect(self.next)
        self.ui.previous.clicked.connect(self.previous)
        self.ui.next_step.clicked.connect(self.tracking)
        self.ui.color1.clicked.connect(self.color1)
        self.ui.color2.clicked.connect(self.color2)
        self.ui.color3.clicked.connect(self.color3)
        self.ui.color4.clicked.connect(self.color4)
        self.ui.color5.clicked.connect(self.color5)
        self.ui.color6.clicked.connect(self.color6)
        self.ui.color7.clicked.connect(self.color7)
        self.ui.color8.clicked.connect(self.color8)
        self.ui.cellDelete.clicked.connect(self.deletecell)
        self.ui.reload_xml.clicked.connect(self.reload_excel)
        self.ui.lineEdit.setPlaceholderText('which img?')
        self.ui.lineEdit.returnPressed.connect(self.check_input)
        ## Make all plots clickable
        self.clickedPen = mkPen('b', width=2)
        self.lastClicked = []
        self.shrink = np.array([7, 70, 7]).T


    def deletecell(self):
        choice = QMessageBox.question(
            self.ui,
            'confirm',
            'Are you sure to detele selected cells? the selected cells will be permantely removed!')
        if choice == QMessageBox.Yes:
            delete_indexes(self.base_direc, self.current, self.lastClicked)
            self.lastClicked = []
            self.w1.clear()
            self.ui.threedView.clear()
            cells_0, unit_vector = read_excels_lines(self.current, self.base_direc)
            unit_vector = unit_vector.split(" ")
            # print(unit_vector)
            unit_vector = list(map(float, unit_vector))
            XZ = points_projection(unit_vector, cells_0[:, 1:4])
            preds_y = cells_0[:, -1]
            preds_y = preds_y.astype(int)
            color = np.array(self.colorlist)
            s1 = ScatterPlotItem(size=10, pen=mkPen(None))
            spots = [{'pos': XZ[i, :], 'data': 1, 'brush': mkColor(self.colorlist[preds_y[i]])} for i in range(np.shape(XZ)[0])]
            s1.addPoints(spots)
            self.w1.addItem(s1)
            s2_color = np.array([color[preds_y[i]][0:3]/255 for i in range(len(preds_y))])
            s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=s2_color, size=5)
            self.ui.threedView.addItem(s2)
            s1.sigClicked.connect(self.clicked)

    def reload_excel(self):
        # self.ui.progressBar.setValue(1)
        self.setProgress(0)
        self.ui.progressBar.show()
        def workerThreadFunc():
            self.ongoing = True
            shape = length_imgs(self.base_direc)
            xml = get_imlist(self.base_direc, '.xml')
            domTree = parse(xml[0])
            rootNode = domTree.documentElement
            Model = rootNode.getElementsByTagName("Model")
            AllSpots = Model[0].getElementsByTagName("AllSpots")
            SpotsInFrame = AllSpots[0].getElementsByTagName("SpotsInFrame")
            ImageData = get_settings(self.base_direc)
            new_ranks = link_lines(self.length, self.base_direc)
            for idx in range(self.length):
                sleep(1)
                reload_excel_from_scratch(self.base_direc, idx, SpotsInFrame, ImageData, shape, new_ranks)
                so.progress_update.emit(idx + 1)
            self.ongoing = False

        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return
        worker = Thread(target=workerThreadFunc, daemon=True)
        worker.start()

    def sort_id(self):
        # self.ui.progressBar.setValue(1)
        self.setProgress(0)
        self.ui.progressBar.show()
        def workerThreadFunc():
            self.ongoing = True
            xml = get_imlist(self.base_direc, '.xml')
            domTree = ps(xml[0])
            dT = judge_blank_spots(domTree)
            time_intervel = float(get_settings(self.base_direc)[0].getAttribute("timeinterval"))
            start_cell_id = 0
            for idx in range(self.length):
                sleep(1)
                dT, start_cell_id = reset_ids(self.base_direc, idx, dT, start_cell_id, time_intervel)
                so.progress_update.emit(idx + 1)
            dT.write(f"{xml[0]}")
            self.ongoing = False

        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return
        worker = Thread(target=workerThreadFunc, daemon=True)
        worker.start()

    def hide_color_buttons(self):
        self.ui.color1.hide()
        self.ui.color2.hide()
        self.ui.color3.hide()
        self.ui.color4.hide()
        self.ui.color5.hide()
        self.ui.color6.hide()
        self.ui.color7.hide()
        self.ui.color8.hide()
        self.ui.cellDelete.hide()

    def color1(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 0)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[0]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)


    def color2(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 1)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[1]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def color３(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 2)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[2]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)


    def color4(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 3)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[3]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array([tuple(item/255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def color5(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 4)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[4]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def color6(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 5)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[5]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def color7(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 6)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[6]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array(
            [tuple(item / 255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def color8(self):
        # choice = QMessageBox.question(
        #     self.ui,
        #     'confirm',
        #     'Are you sure to change the selected points\' color into this one ?')
        # if choice == QMessageBox.Yes:
        refine_excel(self.base_direc, self.current, self.lastClicked, 7)
        for p in self.lastClicked:
            p.setBrush(mkColor(self.colorlist[7]))
        cells_0, _ = read_excels_lines(self.current, self.base_direc)
        cells_0_3d = np.array([tuple(item/255 for item in self.colorlist[int(cells_0[i][-1])][:3]) for i in range(len(cells_0))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=cells_0_3d, size=5)
        self.ui.threedView.clear()
        self.ui.threedView.addItem(s2)

    def setcolor(self):
        colors = []
        for i in range(8):
            colors.append(self.colorlist[i][:3])
        self.ui.color1.setStyleSheet("background-color:rgb{0}".format(colors[0]))
        self.ui.color2.setStyleSheet("background-color:rgb{0}".format(colors[1]))
        self.ui.color3.setStyleSheet("background-color:rgb{0}".format(colors[2]))
        self.ui.color4.setStyleSheet("background-color:rgb{0}".format(colors[3]))
        self.ui.color5.setStyleSheet("background-color:rgb{0}".format(colors[4]))
        self.ui.color6.setStyleSheet("background-color:rgb{0}".format(colors[5]))
        self.ui.color7.setStyleSheet("background-color:rgb{0}".format(colors[6]))
        self.ui.color8.setStyleSheet("background-color:rgb{0}".format(colors[7]))

    def clicked(self, plot, points):
        self.ui.color1.show()
        self.ui.color2.show()
        self.ui.color3.show()
        self.ui.color4.show()
        self.ui.color5.show()
        self.ui.color6.show()
        self.ui.color7.show()
        self.ui.color8.show()
        self.ui.cellDelete.show()
        for p in self.lastClicked:
            if p!=None:
                p.resetPen()
        for i, p in enumerate(points):
            p.setPen(self.clickedPen)
        self.lastClicked = points


    def choosefile(self):
        filePath = QFileDialog.getExistingDirectory(self.ui, "select directory")
        self.base_direc = filePath
        self.set_text()

    def set_text(self):
        self.ui.textEdit.setPlainText(f'{self.base_direc}')

    def showcurrent(self):
        self.ui.currentImage.setPlainText(f'No.{self.current+1} / {self.length} ')

    def setProgress(self, value):
        self.ui.progressBar.setValue(value)

    def clustering(self):
        def workerThreadFunc():
            if ospath.exists(f'{self.base_direc}/tracking'):
                clustered = get_imlist(f'{self.base_direc}/tracking', '.xlsx')
            else:
                clustered = []
            self.ui.progressBar.setValue(len(clustered))
            self.ui.progressBar.show()
            self.ongoing = True

            try:
                pool = Pool(8)  # on 8 processors
                engine = ga_processing(self.base_direc)
                # parameters = [idx for idx in range(len(clustered), self.length)]
                # data_outputs = pool.map(engine, parameters)
                ths = []
                for idx in range(len(clustered), self.length):
                    r = pool.apply_async(engine, args=(idx,))
                    ths.append(r)
                for idx, r in enumerate(ths):
                    r.wait()
                    so.progress_update.emit(idx + 1)
            finally:  # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()

            # for idx in range(len(clustered), self.length):
            #     sleep(1)
            #     # 设置进度值
            #     read_xml(self.base_direc, idx)
            #     so.progress_update.emit(idx+1)
            #
                # self.ui.progressBar.setValue(idx + 1)
            self.ongoing = False
        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return

        worker = Thread(target=workerThreadFunc, daemon=True)
        worker.start()


    def clustering_check(self):
        self.hide_color_buttons()
        if ospath.exists(f"{self.base_direc}/tracking"):
            clustered = get_imlist(f'{self.base_direc}/tracking', '.xlsx')
            if len(clustered) == 0:
                QMessageBox.critical(
                    self.ui,
                    'Wrong',
                    'Please click automatic clustering button first！')
            else:
                self.ui.refresh.show()
                self.clustered_frames = clustered
                if len(clustered) != 1 and self.current != self.length-1:
                    self.ui.next.show()
                cells_0, unit_vector = read_excels_lines(self.current, self.base_direc)
                unit_vector = unit_vector.split(" ")
                unit_vector = list(map(float, unit_vector))
                XZ = points_projection(unit_vector, cells_0[:, 1:4])
                preds_y = cells_0[:, -1]
                preds_y = preds_y.astype(int)
                color = np.array(self.colorlist)
                s1 = ScatterPlotItem(size=10, pen=mkPen(None))
                spots = [{'pos': XZ[i, :], 'data': 1, 'brush': mkColor(self.colorlist[preds_y[i]])} for i in
                         range(np.shape(XZ)[0])]
                s1.addPoints(spots)
                self.w1.addItem(s1)
                s2_color = np.array([color[preds_y[i]][0:3]/255 for i in range(len(preds_y))])
                s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4])/self.shrink, color=s2_color, size=5)
                self.ui.threedView.addItem(s2)
                s1.sigClicked.connect(self.clicked)

        else:
            QMessageBox.critical(
                self.ui,
                'Wrong',
                'Please click automatic clustering button first！')

    def next(self):
        self.lastClicked = []
        self.w1.clear()
        self.ui.threedView.clear()
        self.current += 1
        self.showcurrent()
        self.hide_color_buttons()
        if self.current == len(self.clustered_frames)-1:
            self.ui.next.hide()
        self.ui.previous.show()
        cells_0, unit_vector = read_excels_lines(self.current, self.base_direc)
        unit_vector = unit_vector.split(" ")
        # print(unit_vector)
        unit_vector = list(map(float, unit_vector))
        XZ = points_projection(unit_vector, cells_0[:, 1:4])
        preds_y = cells_0[:, -1]
        preds_y = preds_y.astype(int)
        color = np.array(self.colorlist)
        s1 = ScatterPlotItem(size=10, pen=mkPen(None))
        spots = [{'pos': XZ[i, :], 'data': 1, 'brush': mkColor(self.colorlist[preds_y[i]])} for i in
                 range(np.shape(XZ)[0])]
        s1.addPoints(spots)
        self.w1.addItem(s1)
        s2_color = np.array([color[preds_y[i]][0:3]/255 for i in range(len(preds_y))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=s2_color, size=5)
        self.ui.threedView.addItem(s2)
        s1.sigClicked.connect(self.clicked)

    def can_convert_to_int(self, string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def check_input(self):
        input_img_no = self.ui.lineEdit.text()
        if self.can_convert_to_int(input_img_no):
            input_img_no = int(input_img_no)
            if input_img_no >= 1 and input_img_no <= len(self.clustered_frames):
                self.ui.lineEdit.clear()
                self.goto_which_img(input_img_no-1)
            else:
                QMessageBox.critical(
                    self.ui,
                    'Wrong',
                    'Please input the right number！')
                self.ui.lineEdit.clear()
        else:
            QMessageBox.critical(
                self.ui,
                'Wrong',
                'Please input the right number！')
            self.ui.lineEdit.clear()

    def goto_which_img(self, input_img_no):
        self.lastClicked = []
        self.w1.clear()
        self.ui.threedView.clear()
        self.current = input_img_no
        self.showcurrent()
        self.hide_color_buttons()
        if self.current == len(self.clustered_frames) - 1:
            self.ui.next.hide()
        else:
            self.ui.next.show()
        if self.current == 0:
            self.ui.previous.hide()
        else:
            self.ui.previous.show()
        cells_0, unit_vector = read_excels_lines(self.current, self.base_direc)
        unit_vector = unit_vector.split(" ")
        # print(unit_vector)
        unit_vector = list(map(float, unit_vector))
        XZ = points_projection(unit_vector, cells_0[:, 1:4])
        preds_y = cells_0[:, -1]
        preds_y = preds_y.astype(int)
        color = np.array(self.colorlist)
        s1 = ScatterPlotItem(size=10, pen=mkPen(None))
        spots = [{'pos': XZ[i, :], 'data': 1, 'brush': mkColor(self.colorlist[preds_y[i]])} for i in
                 range(np.shape(XZ)[0])]
        s1.addPoints(spots)
        self.w1.addItem(s1)
        s2_color = np.array([color[preds_y[i]][0:3] / 255 for i in range(len(preds_y))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=s2_color, size=5)
        self.ui.threedView.addItem(s2)
        s1.sigClicked.connect(self.clicked)

    def previous(self):
        self.lastClicked = []
        self.w1.clear()
        self.ui.threedView.clear()
        self.current += -1
        self.showcurrent()
        self.hide_color_buttons()
        if self.current == 0:
            self.ui.previous.hide()
        self.ui.next.show()
        cells_0, unit_vector = read_excels_lines(self.current, self.base_direc)
        unit_vector = unit_vector.split(" ")
        # print(unit_vector)
        unit_vector = list(map(float, unit_vector))
        XZ = points_projection(unit_vector, cells_0[:, 1:4])
        preds_y = cells_0[:, -1]
        preds_y = preds_y.astype(int)
        color = np.array(self.colorlist)
        s1 = ScatterPlotItem(size=10, pen=mkPen(None))
        spots = [{'pos': XZ[i, :], 'data': 1, 'brush': mkColor(self.colorlist[preds_y[i]])} for i in
                 range(np.shape(XZ)[0])]
        s1.addPoints(spots)
        self.w1.addItem(s1)
        s2_color = np.array([color[preds_y[i]][0:3]/255 for i in range(len(preds_y))])
        s2 = GLScatterPlotItem(pos=np.array(cells_0[:, 1:4]) / self.shrink, color=s2_color, size=5)
        self.ui.threedView.addItem(s2)
        s1.sigClicked.connect(self.clicked)

    def tracking(self):
        if len(self.clustered_frames) != self.length:
            if len(self.clustered_frames) > self.length:
                QMessageBox.warning(
                    self.window,
                    'Warning', 'It seems like you have opened an excel in tracking directory, please close it')
            else:
                QMessageBox.critical(
                    self.ui,
                    'Wrong',
                    '8 line clustering is not finished, please click automatic clustering button')
        else:
            self.tracking = Tracking(self.base_direc, self.length)
            self.tracking.ui.show()
            self.ui.close()

class Detection(QMainWindow):
    def __init__(self):
        so2.progress_update.connect(self.setProgress)
        t2.terminate.connect(self.closeui)
        self.ui = QUiLoader().load('detection.ui')
        self.ui.toolButton.clicked.connect(self.choosefile)
        self.ui.detect.hide()
        self.ui.progressBar.hide()
        self.ui.detect.clicked.connect(self.cell_detect)
        self.ongoing = False

    def closeui(self):
        QMessageBox.information(
            self.ui,
            'successful',
            'Detection is completed! Congrats!')
        self.ui.close()

    def setProgress(self,value):
        self.ui.progressBar.setValue(value)

    def set_text(self):
        self.ui.textEdit.setPlainText(f'{self.base_direc}')

    def choosefile(self):
        self.base_direc = QFileDialog.getExistingDirectory(self.ui, "select directory")
        self.set_text()
        self.length = length_imgs(self.base_direc)[0]
        self.ui.progressBar.setRange(0, self.length)
        self.ui.detect.show()

    def cell_detect(self):
        self.ui.progressBar.show()
        def workerThreadFunc():
            dev = device("cuda" if cuda.is_available() else "cpu")
            # check_point = load(r'./u_net_cell_best_new_zdouble.pth')
            if self.ui.radioButton.isChecked():
                model = UNet()
                model.cuda(dev)
                check_point = load(r'./unet_2d_best_goh.pth')
            else:

                print('This is 3d unet')
                model = unet3d()
                model.cuda(dev)
                check_point = load(r'./3d_unet_best_goh.pth')

            model.load_state_dict(check_point['state_dict'])
            model.eval()
            img_path = get_imlist(self.base_direc)[0]
            imgs = io.imread(img_path)
            whether_transpose = False
            if np.shape(imgs)[-2] < np.shape(imgs)[-1]:
                imgs = imgs.transpose(0, 1, 3, 2)
                whether_transpose = True
            create_blank_xml(self.base_direc, np.shape(imgs))
            xml = get_imlist(self.base_direc, '.xml')
            domTree = ps(xml[0])
            time_intervel = float(get_settings(self.base_direc)[0].getAttribute("timeinterval"))
            cell_id = 0
            self.ongoing = True
            for idx in range(self.length):
                sleep(1)
                # 设置进度值
                if self.ui.radioButton.isChecked():
                    commons, mitotics = test_model(model, self.base_direc, idx, imgs, dev, whether_transpose)
                else:
                    commons, mitotics = test_3d(model, self.base_direc, idx, imgs, dev, whether_transpose)
                cell_id, domTree = xml_from_centroids(idx, commons, mitotics, domTree, cell_id,
                                                      time_intervel, self.base_direc)
                so2.progress_update.emit(idx+1)
                # self.ui.progressBar.setValue(idx + 1)

            domTree.write(f"{xml[0]}")
            self.ongoing = False
            t2.terminate.emit()

        if self.ongoing:
            QMessageBox.warning(
                self.window,
                'Warning', 'In progress, please wait')
            return
        worker = Thread(target=workerThreadFunc, daemon=True)
        worker.start()


class Stats(QMainWindow):
    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('main.ui')

        self.ui.toolButton.clicked.connect(self.choosefile)

    def choosefile(self):
        filePath = QFileDialog.getExistingDirectory(self.ui, "select directory")
        self.clustering = Clustering(filePath)
        # 显示新窗口
        self.clustering.ui.show()
        # 关闭自己
        self.ui.close()

class Entrance(QMainWindow):
    def __init__(self):
        self.ui = QUiLoader().load('entrance.ui')
        self.ui.cell_detection.clicked.connect(self.detection)
        self.ui.cell_tracking.clicked.connect(self.tracking)

    def tracking(self):
        self.stats = Stats()
        self.stats.ui.show()
        self.ui.close()

    def detection(self):
        self.detect = Detection()
        self.detect.ui.show()
        self.ui.close()

if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon('Ritsumeikan.png'))
    entrance = Entrance()
    entrance.ui.show()
    sys.exit(app.exec_())