import sys, math
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QLabel, QTextEdit,
    QMessageBox, QGroupBox, QFileDialog, QAbstractItemView, QProgressBar
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
QProgressBar { background-color: #eee; border: none; height: 4px; }
QProgressBar::chunk { background-color: #4a90d9; }
"""


def calibrate(repers):
    n = len(repers)
    if n < 2:
        return None, "Kamida 2 ta reper kerak"
    dists = np.array([r["distance"] for r in repers], dtype=float)
    design = np.array([r["design"] for r in repers], dtype=float)
    measured = np.array([r["measured"] for r in repers], dtype=float)
    diff = measured - design
    A = np.column_stack([np.ones(n), dists])
    try:
        coeffs, *_ = np.linalg.lstsq(A, diff, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    except np.linalg.LinAlgError:
        a, b = 0.0, 0.0
    corrected = measured - (a + b * dists)
    resid = diff - (a + b * dists)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((diff - np.mean(diff)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else (1.0 if ss_res < 1e-15 else 0.0)
    dof = max(n - 2, 0)
    sigma = math.sqrt(ss_res / dof) if dof > 0 else 0.0
    var_d = float(np.sum((dists - np.mean(dists)) ** 2))
    if sigma > 0 and n > 2 and var_d > 1e-15:
        se_a = sigma * math.sqrt(float(np.sum(dists ** 2)) / (n * var_d))
        se_b = sigma / math.sqrt(var_d)
    else:
        se_a = 0.0
        se_b = 0.0
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 and float(np.var(diff, ddof=1)) > 0 else 0.0
    return {
        "diff": diff, "corrected": corrected, "resid": resid,
        "a": a, "b": b, "r2": r2, "sigma": sigma,
        "se_a": se_a, "se_b": se_b, "n": n,
        "design": design, "measured": measured, "dists": dists,
        "mean_diff": mean_diff, "std_diff": std_diff
    }, None


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
        self.table = QTableWidget(0, 4)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["Reper", "Masofa (m)", "Dizayn (m)", "O'lchangan (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        ml.addWidget(self.table)
        for nm, ds, de, me in [("R-1", "0", "100.000", "100.003"), ("R-2", "500", "105.000", "104.995"), ("R-3", "1000", "110.000", "110.008")]:
            self.add_row_data(nm, ds, de, me)

    def add_row_data(self, name, dist, design, measured):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setRowHeight(r, 30)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(dist))
        self.table.setItem(r, 2, QTableWidgetItem(design))
        self.table.setItem(r, 3, QTableWidgetItem(measured))

    def add_row(self):
        r = self.table.rowCount()
        self.add_row_data(f"R-{r + 1}", str(r * 500) if r > 0 else "0", "100.000", "100.000")

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
            if not all(items):
                continue
            try:
                data.append({
                    "name": items[0].text().strip(),
                    "distance": float(items[1].text()),
                    "design": float(items[2].text()),
                    "measured": float(items[3].text())
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
            QMessageBox.warning(self, "Xatolik", "Ma'lumotlarni tekshiring.\nBarcha maydonlar to'g'ri to'ldirilganiga ishonch hosil qiling.")
            return
        if len(data) < 2:
            QMessageBox.warning(self, "Xatolik", "Kamida 2 ta reper kerak!")
            return
        dists = [r["distance"] for r in data]
        if all(d == dists[0] for d in dists):
            QMessageBox.warning(self, "Ogohlantirish", "Masofalar bir xil. Kalibrlash faqat o'rtacha farqni hisoblaydi.")
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
        L.append("=" * 72)
        L.append("  REPER KALIBRASH NATIJALARI")
        L.append("=" * 72)
        L.append("")
        L.append("--- REPERLAR TAHLILI ---")
        L.append("")
        L.append(f"  {'Reper':>6} {'Masofa':>10} {'Dizayn':>12} {"O'lchov":>12} {'Farq':>8} {'Tuzatilgan':>14}")
        L.append("  " + "-" * 64)
        for i, rp in enumerate(data):
            d = res["diff"][i]
            corr = res["corrected"][i]
            L.append(f"  {rp['name']:>6} {rp['distance']:>10.2f} {rp['design']:>12.4f} {rp['measured']:>12.4f} {d*1000:>7.2f} {corr:>14.4f}")
        L.append("")
        L.append("--- STATISTIK TAHLIL ---")
        L.append("")
        L.append(f"  Reperlar soni:            {res['n']}")
        L.append(f"  Ortacha farq:             {res['mean_diff']*1000:.2f} mm")
        L.append(f"  Farqlarning std:          {res['std_diff']*1000:.2f} mm" if res['std_diff'] > 0 else f"  Farqlarning std:          {'—':>8}")
        L.append(f"  Doimiy xato (a):          {res['a']*1000:+.4f} mm")
        if res['n'] > 2:
            L.append(f"  Masofaviy xato (b):       {res['b']*1e6:+.4f} mm/km")
        else:
            L.append(f"  Masofaviy xato (b):       {'— (2 ta reper)':>15}")
        if res['sigma'] > 0:
            L.append(f"  Sigma0:                   {res['sigma']*1000:.4f} mm")
        L.append(f"  R2 (determinatsiya):      {res['r2']:.6f}")
        L.append("")
        L.append("--- KALIBRASH TENGLAMASI ---")
        L.append("")
        if res['n'] > 2:
            L.append(f"  H_tuzatilgan = H_olchangan - ({res['a']*1000:+.4f} mm + {res['b']*1e6:+.4f} mm/km * D)")
        else:
            L.append(f"  H_tuzatilgan = H_olchangan - ({res['a']*1000:+.4f} mm)  (faqat doimiy xato)")
        L.append("")
        L.append("--- QOLDIQLAR ---")
        L.append("")
        for i, rp in enumerate(data):
            L.append(f"  {rp['name']:>6}: v = {res['resid'][i]*1000:+8.2f} mm")
        L.append("")
        L.append("--- XULOSA ---")
        L.append("")
        max_err = max(abs(res["diff"])) * 1000
        if res['n'] == 2:
            L.append(f"  Eng katta farq: {max_err:.2f} mm")
            L.append("  Eslatma: 2 ta reper bilan faqat doimiy xato aniqlanadi.")
        elif max_err < 3:
            L.append(f"  Eng katta farq: {max_err:.2f} mm")
            L.append("  Repyerlar yaroqli.")
        elif max_err < 10:
            L.append(f"  Eng katta farq: {max_err:.2f} mm")
            L.append("  Repyerlar qoniqarli. Farqlar kuzatilmoqda.")
        else:
            L.append(f"  Eng katta farq: {max_err:.2f} mm")
            L.append("  Repyerlarni qayta korib chiqish tavsiya etiladi.")
        L.append("")
        L.append("-" * 72)
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
            self.rw.add_row_data("R-1", "0", "100.000", "100.003")
            self.rw.add_row_data("R-2", "500", "105.000", "104.995")
            self.rw.add_row_data("R-3", "1000", "110.000", "110.008")
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
