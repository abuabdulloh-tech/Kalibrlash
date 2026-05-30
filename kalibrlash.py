import sys, math
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QLabel, QTextEdit,
    QMessageBox, QGroupBox, QFileDialog, QAbstractItemView, QProgressBar,
    QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

STYLE = """
QMainWindow, QWidget { background-color: #fff; color: #222; }
QGroupBox {
    font-size: 14px; font-weight: bold;
    border: 1px solid #ccc; border-radius: 6px;
    margin-top: 16px; padding: 20px 14px 14px 14px; background-color: #f8f8f8;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #333; }
QTableWidget {
    background-color: #fff; alternate-background-color: #fafafa;
    gridline-color: #e0e0e0; border: 1px solid #ccc; font-size: 13px;
}
QTableWidget::item { padding: 6px 10px; }
QTableWidget::item:selected { background-color: #e8f0fe; color: #000; }
QHeaderView::section {
    background-color: #f0f0f0; color: #333; padding: 8px 10px;
    border: none; border-bottom: 1px solid #ddd;
    font-weight: bold; font-size: 12px;
}
QPushButton {
    background-color: #f0f0f0; color: #222;
    border: 1px solid #ccc; border-radius: 4px;
    padding: 8px 20px; font-size: 13px;
}
QPushButton:hover { background-color: #e0e0e0; }
QPushButton:pressed { background-color: #d0d0d0; }
QPushButton:disabled { background-color: #f8f8f8; color: #999; border-color: #ddd; }
QTextEdit {
    background-color: #fff; color: #222;
    border: 1px solid #ccc; border-radius: 4px;
    padding: 10px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;
}
QComboBox {
    background-color: #fff; color: #222;
    border: 1px solid #bbb; border-radius: 3px;
    padding: 4px 6px; font-size: 12px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #fff; color: #222;
    selection-background-color: #e8f0fe;
}
QProgressBar { background-color: #eee; border: none; height: 4px; }
QProgressBar::chunk { background-color: #4a90d9; }
"""


def network_adjust(repers):
    n = len(repers)
    if n < 2:
        return None, "Kamida 2 ta reper kerak"

    segs = n - 1
    dists = np.array([r["distance"] for r in repers], dtype=float)
    design = np.array([r["design"] for r in repers], dtype=float)
    measured = np.array([r["measured"] for r in repers], dtype=float)
    fixed = [r.get("fixed", False) for r in repers]

    dh_meas = np.array([measured[i+1] - measured[i] for i in range(segs)])
    dh_design = np.array([design[i+1] - design[i] for i in range(segs)])
    dh_diff = dh_meas - dh_design

    seg_dists = np.array([dists[i+1] - dists[i] for i in range(segs)])
    sigma_per_km = 2.0
    weights = np.array([1.0 / (sigma_per_km * math.sqrt(d / 1000.0) / 1000.0) ** 2 if d > 0 else 1e12 for d in seg_dists])

    fixed_idx = [i for i in range(n) if fixed[i]]
    adj_idx = [i for i in range(n) if not fixed[i]]
    nu = len(adj_idx)
    no = segs
    if nu == 0:
        return None, "Kamida 1 ta sozlanadigan reper kerak"
    if no < 1:
        return None, "O'lchovlar yo'q"

    idx_map = {orig: new for new, orig in enumerate(adj_idx)}
    fixed_height = {i: design[i] for i in fixed_idx}

    A = np.zeros((no, nu))
    l = np.zeros(no)
    for i in range(segs):
        rhs = dh_meas[i]
        if i in fixed_height:
            rhs -= -fixed_height[i]
        if i + 1 in fixed_height:
            rhs -= fixed_height[i + 1]
        rhs = dh_meas[i]
        if i in fixed_idx:
            rhs -= -design[i]
        if i + 1 in fixed_idx:
            rhs -= design[i + 1]
        rhs = dh_meas[i]
        s = 0
        if i in fixed_idx:
            s -= design[i]
        if i + 1 in fixed_idx:
            s += design[i + 1]
        rhs = dh_meas[i] - s

        if i in adj_idx:
            A[i, idx_map[i]] = -1
        if i + 1 in adj_idx:
            A[i, idx_map[i + 1]] = 1
        l[i] = rhs

    P = np.diag(weights)
    try:
        N = A.T @ P @ A
        b = A.T @ P @ l
        U, S, Vt = np.linalg.svd(N)
        rcond = max(N.shape) * np.finfo(float).eps * max(S)
        S_inv = np.array([1 / s if s > rcond else 0 for s in S])
        N_inv = Vt.T @ np.diag(S_inv) @ U.T
        x = N_inv @ b
    except np.linalg.LinAlgError:
        return None, "Matritsani yechib bo'lmadi"

    v = A @ x - l
    dof = no - nu
    s02 = v.T @ P @ v / dof if dof > 0 else 0
    s0 = math.sqrt(s02) if s02 > 0 else 0

    Qxx = N_inv.copy()
    Qvv = np.diag(1 / weights) - A @ Qxx @ A.T
    w_stats = np.zeros(no)
    for i in range(no):
        qv = Qvv[i, i]
        w_stats[i] = abs(v[i] / math.sqrt(qv)) if qv > 0 and s0 > 0 else 0

    adj_heights = design.copy()
    for orig, new in idx_map.items():
        adj_heights[orig] = float(x[new])
    stds = np.zeros(n)
    for orig, new in idx_map.items():
        stds[orig] = math.sqrt(Qxx[new, new]) * s0 if s0 > 0 else 0

    outlier_idx = [i for i in range(no) if w_stats[i] > 3.29]

    return {
        "adj_heights": adj_heights, "stds": stds,
        "s0": s0, "dof": dof, "n": n, "no": no, "nu": nu,
        "v": v, "w_stats": w_stats, "seg_dists": seg_dists,
        "dh_meas": dh_meas, "dh_design": dh_design,
        "dh_diff": dh_diff, "outlier_idx": outlier_idx,
        "design": design, "measured": measured, "dists": dists,
        "fixed": fixed, "seg_weights": weights
    }, None


def calibrate(repers):
    return network_adjust(repers)


class Worker(QThread):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, repers):
        super().__init__()
        self.repers = repers

    def run(self):
        res, err = calibrate(self.repers)
        if err:
            self.fail.emit(err)
        else:
            self.done.emit(res)


class ReperWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup()

    def setup(self):
        ml = QVBoxLayout(self)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(8)
        lbl = QLabel("Ma'lumotlarni kiriting:")
        lbl.setStyleSheet("font-size: 13px; color: #555;")
        ml.addWidget(lbl)
        br = QHBoxLayout()
        ab = QPushButton("+ Qo'shish")
        ab.clicked.connect(self.add_row)
        br.addWidget(ab)
        rb = QPushButton("- O'chirish")
        rb.clicked.connect(self.remove_selected)
        br.addWidget(rb)
        br.addStretch()
        ml.addLayout(br)
        self.table = QTableWidget(0, 5)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["Reper", "Masofa (m)", "Dizayn (m)", "O'lchangan (m)", "Holat"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        ml.addWidget(self.table)
        for nm, ds, de, me, fx in [("R-1", "0", "100.000", "100.003", True), ("R-2", "500", "105.000", "104.995", False), ("R-3", "1000", "110.000", "110.008", False)]:
            self.add_row_data(nm, ds, de, me, fx)

    def add_row_data(self, name, dist, design, measured, fixed=False):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setRowHeight(r, 30)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(dist))
        self.table.setItem(r, 2, QTableWidgetItem(design))
        self.table.setItem(r, 3, QTableWidgetItem(measured))
        cb = QComboBox()
        cb.addItems(["Sozlanadi", "Turg'un"])
        cb.setCurrentIndex(1 if fixed else 0)
        self.table.setCellWidget(r, 4, cb)

    def add_row(self):
        r = self.table.rowCount()
        prev = f"{r * 500}" if r > 0 else "0"
        self.add_row_data(f"R-{r + 1}", prev, "100.000", "100.000", False)

    def remove_selected(self):
        rows = set()
        for item in self.table.selectedItems():
            rows.add(item.row())
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)
        self.rename_all()

    def rename_all(self):
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                it.setText(f"R-{r + 1}")

    def get_data(self):
        data = []
        for r in range(self.table.rowCount()):
            items = [self.table.item(r, c) for c in range(4)]
            cb = self.table.cellWidget(r, 4)
            if not all(items) or not cb:
                continue
            try:
                data.append({
                    "name": items[0].text().strip(),
                    "distance": float(items[1].text()),
                    "design": float(items[2].text()),
                    "measured": float(items[3].text()),
                    "fixed": cb.currentText() == "Turg'un"
                })
            except ValueError:
                pass
        return data


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reper Kalibrlash")
        self.setMinimumSize(1300, 900)
        self.worker = None
        self.last = None
        self.build()

    def build(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)
        ml.setContentsMargins(20, 16, 20, 16)
        ml.setSpacing(12)

        hdr = QHBoxLayout()
        t = QLabel("Reper Kalibrlash")
        t.setStyleSheet("font-size: 22px; font-weight: bold; color: #333;")
        hdr.addWidget(t)
        hdr.addStretch()
        ml.addLayout(hdr)

        inp = QGroupBox("Reper ma'lumotlari")
        il = QVBoxLayout(inp)
        self.rw = ReperWidget()
        il.addWidget(self.rw)
        ml.addWidget(inp)

        br = QHBoxLayout()
        br.setSpacing(8)
        self.rb = QPushButton("Hisoblash")
        self.rb.setMinimumHeight(40)
        self.rb.setStyleSheet("background-color: #4a90d9; color: #fff; font-size: 14px; border: none; padding: 10px 40px; border-radius: 4px;")
        self.rb.clicked.connect(self.calc)
        br.addStretch()
        br.addWidget(self.rb)
        sb = QPushButton("Saqlash")
        sb.clicked.connect(self.save)
        br.addWidget(sb)
        cb = QPushButton("Tozalash")
        cb.clicked.connect(self.clear)
        br.addWidget(cb)
        br.addStretch()
        ml.addLayout(br)

        self.pr = QProgressBar()
        self.pr.setRange(0, 0)
        self.pr.hide()
        ml.addWidget(self.pr)

        rg = QGroupBox("Kalibrlash natijalari")
        rl = QVBoxLayout(rg)
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(260)
        rl.addWidget(self.out)
        ml.addWidget(rg)

    def calc(self):
        data = self.rw.get_data()
        if not data:
            QMessageBox.warning(self, "Xatolik", "Ma'lumotlarni tekshiring.")
            return
        if len(data) < 2:
            QMessageBox.warning(self, "Xatolik", "Kamida 2 ta reper kerak!")
            return
        if sum(1 for r in data if r.get("fixed")) == 0:
            QMessageBox.warning(self, "Xatolik", "Kamida 1 ta turg'un reper belgilang!")
            return
        self.pr.show()
        self.rb.setEnabled(False)
        self.out.setText("Hisoblanmoqda...")
        self.worker = Worker(data)
        self.worker.done.connect(self.on_done)
        self.worker.fail.connect(self.on_fail)
        self.worker.start()

    def on_done(self, res):
        self.pr.hide()
        self.rb.setEnabled(True)
        if res is None:
            self.out.setText("Xatolik: hisoblash natijasi yo'q.")
            return
        self.last = res
        data = self.rw.get_data()
        self.out.setPlainText(self.fmt(res, data))

    def on_fail(self, msg):
        self.pr.hide()
        self.rb.setEnabled(True)
        self.out.setText(f"XATO: {msg}")
        QMessageBox.critical(self, "Xatolik", msg)

    def fmt(self, res, data):
        L = []
        L.append("=" * 78)
        L.append("  REPER KALIBRASH NATIJALARI  (tarmoq tenglashtirish)")
        L.append("=" * 78)
        L.append("")

        L.append("--- REPERLAR ---")
        L.append("")
        L.append(f"  {'Reper':>6} {'Masofa':>10} {'Dizayn':>12} {'O\'lchov':>12} {'Farq(mm)':>10} {'Tuzatilgan':>14} {'Std(mm)':>10} {'Holat':>10}")
        L.append("  " + "-" * 76)
        for i, rp in enumerate(data):
            dh = (res["measured"][i] - res["design"][i]) * 1000
            std = res["stds"][i] * 1000 if res["stds"][i] > 0 else 0
            status = "TURG'UN" if res["fixed"][i] else "SOZLANDI"
            L.append(f"  {rp['name']:>6} {res['dists'][i]:>10.2f} {res['design'][i]:>12.4f} {res['measured'][i]:>12.4f} {dh:>9.2f} {res['adj_heights'][i]:>14.4f} {std:>9.2f}  {status:>10}")
        L.append("")

        L.append("--- SEGMENTLAR (qo'shni reperlar orasidagi farqlar) ---")
        L.append("")
        L.append(f"  {'Segment':>10} {'Masofa':>10} {'Dizayn(m)':>12} {'O\'lchov(m)':>12} {'Farq(mm)':>10} {'Qoldiq(mm)':>12} {'w-stat':>8} {'Holat':>12}")
        L.append("  " + "-" * 78)
        for i in range(res["no"]):
            w = res["w_stats"][i]
            outlier = "QO'POL!" if w > 3.29 else "OK"
            L.append(f"  {data[i]['name']:>4}-{data[i+1]['name']:<4} {res['seg_dists'][i]:>10.2f} {res['dh_design'][i]:>12.4f} {res['dh_meas'][i]:>12.4f} {res['dh_diff'][i]*1000:>9.2f} {res['v'][i]*1000:>11.2f} {w:>7.2f}  {outlier:>12}")
        L.append("")

        L.append("--- STATISTIK TAHLIL ---")
        L.append("")
        L.append(f"  Reperlar soni:              {res['n']}")
        L.append(f"  Segmentlar soni:            {res['no']}")
        L.append(f"  Sozlanadigan nuqtalar:      {res['nu']}")
        L.append(f"  Erkinlik darajasi:          {res['dof']}")
        L.append(f"  Sigma0:                     {res['s0']*1000:.4f} mm")
        if res["outlier_idx"]:
            L.append(f"  Qo'pol xatolar:             {len(res['outlier_idx'])} ta segmentda")
        else:
            L.append(f"  Qo'pol xatolar:             aniqlanmadi")
        L.append("")

        if res["outlier_idx"]:
            L.append("--- QO'POL XATOLI SEGMENTLAR ---")
            L.append("")
            for i in res["outlier_idx"]:
                w = res["w_stats"][i]
                L.append(f"  {data[i]['name']}-{data[i+1]['name']}:  w = {w:.2f}  (chegara: 3.29)")
                L.append(f"    O'lchov farqi: {res['dh_meas'][i]:.4f}m, Dizayn farqi: {res['dh_design'][i]:.4f}m")
                L.append(f"    Farq: {res['dh_diff'][i]*1000:.2f}mm, Qoldiq: {res['v'][i]*1000:.2f}mm")
            L.append("")

        L.append("--- KALIBRASH XULOSASI ---")
        L.append("")
        all_dh = [abs(res["dh_diff"][i]) * 1000 for i in range(res["no"])]
        max_seg = max(all_dh) if all_dh else 0
        max_pt = max([abs((res["measured"][i] - res["design"][i]) * 1000) for i in range(res["n"])])
        L.append(f"  Eng katta segment farqi:    {max_seg:.2f} mm")
        L.append(f"  Eng katta nuqta farqi:      {max_pt:.2f} mm")
        L.append(f"  Turg'un reperlar:           {sum(res['fixed'])} ta")
        L.append(f"  Sozlanadigan reperlar:      {res['nu']} ta")
        L.append("")
        if res["outlier_idx"]:
            L.append("  DIQQAT: Qo'pol xatolar bor. Turg'un reperlarni tekshiring.")
        elif res["s0"] < 2:
            L.append("  Tarmoq yaxshi. Reperlar barqaror.")
        elif res["s0"] < 5:
            L.append("  Tarmoq qoniqarli. Kichik siljishlar bor.")
        else:
            L.append("  Tarmoqda muammo bor. Reperlar holatini tekshiring.")
        L.append("")
        L.append("-" * 78)
        L.append("  GNU Gama mantig'ida tarmoq tenglashtirish | EKKU")
        L.append("-" * 78)
        return "\n".join(L)

    def save(self):
        if not self.last:
            QMessageBox.warning(self, "Ogohlantirish", "Avval hisoblashni bajarib oling!")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Natijani saqlash", "kalibrlash_natijasi.txt", "Matn (*.txt)")
        if p:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(self.out.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Xatolik", f"Faylga yozishda xatolik: {e}")

    def clear(self):
        r = QMessageBox.question(self, "Tozalash", "Barcha ma'lumotlarni tozalaysizmi?",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes:
            while self.rw.table.rowCount():
                self.rw.table.removeRow(0)
            self.rw.add_row_data("R-1", "0", "100.000", "100.003", True)
            self.rw.add_row_data("R-2", "500", "105.000", "104.995", False)
            self.rw.add_row_data("R-3", "1000", "110.000", "110.008", False)
            self.out.clear()
            self.last = None


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
