import sys, os, math
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QPushButton, QHeaderView, QLabel, QComboBox, QTextEdit,
    QMessageBox, QGroupBox, QDoubleSpinBox, QSpinBox, QFileDialog,
    QTabWidget, QSplitter, QGridLayout, QFrame, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

STYLE = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
}
QGroupBox {
    font-size: 12px; font-weight: bold;
    border: 1px solid #30363d; border-radius: 8px;
    margin-top: 16px; padding: 20px 16px 16px 16px;
    background-color: #161b22; color: #e6edf3;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; padding: 0 10px;
    color: #58a6ff;
}
QTableWidget {
    background-color: #0d1117; alternate-background-color: #161b22;
    gridline-color: #30363d; border: 1px solid #30363d;
    border-radius: 6px; font-size: 14px;
}
QTableWidget::item { padding: 10px 14px; }
QTableWidget::item:selected { background-color: #1f6feb; color: #ffffff; }
QHeaderView::section {
    background-color: #21262d; color: #e6edf3;
    padding: 12px 14px; border: none; border-bottom: 2px solid #30363d;
    font-weight: bold; font-size: 13px;
}
QPushButton {
    background-color: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 24px; font-size: 13px; font-weight: bold;
}
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton:pressed { background-color: #161b22; }
QPushButton:disabled { background-color: #0d1117; color: #484f58; border-color: #21262d; }
QTextEdit {
    background-color: #0d1117; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;
}
QComboBox {
    background-color: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    padding: 8px 12px; font-size: 14px; font-weight: bold;
}
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView {
    background-color: #21262d; color: #e6edf3;
    selection-background-color: #1f6feb; font-size: 13px;
}
QDoubleSpinBox, QSpinBox {
    background-color: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    padding: 8px 12px; font-size: 14px; font-weight: bold;
}
QDoubleSpinBox:disabled, QSpinBox:disabled {
    background-color: #0d1117; color: #484f58; border-color: #21262d;
}
QTabWidget::pane {
    border: 1px solid #30363d; border-radius: 6px;
    background-color: #0d1117; padding: 8px;
}
QTabBar::tab {
    background-color: #161b22; color: #8b949e;
    padding: 12px 32px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-size: 14px; font-weight: bold;
    border: 1px solid #30363d; border-bottom: none;
}
QTabBar::tab:selected { background-color: #0d1117; color: #58a6ff; border-color: #30363d; }
QTabBar::tab:hover:!selected { background-color: #21262d; color: #e6edf3; }
QProgressBar {
    background-color: #21262d; border: none; border-radius: 4px;
    text-align: center; color: #e6edf3; height: 6px;
}
QProgressBar::chunk { background-color: #58a6ff; border-radius: 4px; }
QFrame#card {
    background-color: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px;
}
"""


def adjust_network(points, observations):
    adj_ids = [p["id"] for p in points if p["type"] == "adjusted"]
    fixed = {p["id"]: p["height"] for p in points if p["type"] == "fixed" and p["height"] is not None}
    nu = len(adj_ids)
    no = len(observations)
    if nu == 0:
        return None, "Kamida bitta sozlanadigan nuqta kerak"
    if no < nu:
        return None, f"O'lchovlar soni ({no}) noma'lumlar sonidan ({nu}) kam bo'lmasligi kerak"
    idxi = {pid: i for i, pid in enumerate(adj_ids)}
    A = np.zeros((no, nu))
    l = np.zeros((no, 1))
    P = np.zeros((no, no))
    for i, o in enumerate(observations):
        f, t = o["from"], o["to"]
        w = 1.0 / (o["stdev"] / 1000.0) ** 2
        P[i, i] = w
        rhs = o["val"]
        if f in fixed:
            rhs += fixed[f]
        if t in adj_ids:
            A[i, idxi[t]] = 1
        if f in adj_ids:
            A[i, idxi[f]] = -1
        if t in fixed:
            rhs -= fixed[t]
        l[i, 0] = rhs
    try:
        N = A.T @ P @ A
        b = A.T @ P @ l
        x = np.linalg.solve(N, b)
    except np.linalg.LinAlgError:
        return None, "Matritsani yechib bo'lmadi. Ma'lumotlarni tekshiring."
    v = A @ x - l
    dof = no - nu
    s02 = (v.T @ P @ v)[0, 0] / dof if dof > 0 else 0
    s0 = math.sqrt(s02) if s02 > 0 else 0
    Qxx = np.linalg.inv(N)
    adj = {}
    stds = {}
    for i, pid in enumerate(adj_ids):
        adj[pid] = float(x[i, 0])
        stds[pid] = math.sqrt(Qxx[i, i]) * s0 if s0 > 0 else 0
    resid = [float(v[i, 0]) for i in range(no)]
    w_obs = []
    for i, o in enumerate(observations):
        Qvv = 1 / P[i, i] - A[i:i+1] @ Qxx @ A[i:i+1].T
        wval = float(v[i] / math.sqrt(float(Qvv[0, 0]))) if float(Qvv[0, 0]) > 0 else 0
        w_obs.append(abs(wval))
    return {
        "adjusted": adj, "stds": stds, "s0": s0, "residuals": resid,
        "n_obs": no, "n_unknowns": nu, "dof": dof, "Qxx": Qxx,
        "w_obs": w_obs, "A": A, "l": l, "P": P, "x": x, "v": v, "adj_ids": adj_ids,
        "fixed": fixed, "observations": observations
    }, None


class Worker(QThread):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, points, obs):
        super().__init__()
        self.points = points
        self.obs = obs

    def run(self):
        res, err = adjust_network(self.points, self.obs)
        if err:
            self.fail.emit(err)
        else:
            self.done.emit(res)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reper Kalibrlash")
        self.setMinimumSize(1360, 920)
        self.worker = None
        self.last = None
        self.build()

    def build(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)
        ml.setContentsMargins(24, 24, 24, 24)
        ml.setSpacing(16)

        hdr = QHBoxLayout()
        t = QLabel("Reper Kalibrlash")
        t.setStyleSheet("font-size: 28px; font-weight: bold; color: #58a6ff;")
        hdr.addWidget(t)
        hdr.addStretch()
        sub = QLabel("nivelirlash tarmog'ini tenglashtirish")
        sub.setStyleSheet("font-size: 13px; color: #8b949e; padding-top: 8px;")
        hdr.addWidget(sub)
        ml.addLayout(hdr)

        tabs = QTabWidget()
        ml.addWidget(tabs)
        self.pt = QWidget()
        self.ot = QWidget()
        tabs.addTab(self.pt, "Nuqtalar")
        tabs.addTab(self.ot, "Nivelirlash")
        self.build_ptab()
        self.build_otab()

        br = QHBoxLayout()
        br.setSpacing(12)
        self.rb = QPushButton("Hisoblash")
        self.rb.setMinimumHeight(48)
        self.rb.setStyleSheet("""
            QPushButton {
                background-color: #238636; color: #ffffff;
                border: none; border-radius: 8px;
                font-size: 17px; font-weight: bold; padding: 12px 56px;
            }
            QPushButton:hover { background-color: #2ea043; }
            QPushButton:pressed { background-color: #1a6b2f; }
            QPushButton:disabled { background-color: #21262d; color: #484f58; }
        """)
        self.rb.clicked.connect(self.calc)
        br.addStretch()
        br.addWidget(self.rb)
        eb = QPushButton("Namuna")
        eb.clicked.connect(self.example)
        eb.setStyleSheet("background-color: #1f6feb; border-color: #1f6feb; color: #fff;")
        br.addWidget(eb)
        xb = QPushButton("XML")
        xb.clicked.connect(self.export_xml)
        br.addWidget(xb)
        sb = QPushButton("Saqlash")
        sb.clicked.connect(self.export_txt)
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

        rg = QGroupBox("Natijalar")
        rl = QVBoxLayout(rg)
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(220)
        rl.addWidget(self.out)
        ml.addWidget(rg)

    def build_ptab(self):
        l = QVBoxLayout(self.pt)
        l.setContentsMargins(12, 12, 12, 12)
        info = QLabel("Fiksatsiyalangan nuqtalarga balandlik kiriting. Sozlanadigan nuqtalar balandligi hisoblanadi.")
        info.setStyleSheet("color: #8b949e; font-size: 13px; padding: 4px 0;")
        l.addWidget(info)
        br = QHBoxLayout()
        ab = QPushButton("+ Qo'shish")
        ab.clicked.connect(self.apr)
        br.addWidget(ab)
        br.addStretch()
        l.addLayout(br)
        self.ptt = QTableWidget(0, 3)
        self.ptt.setAlternatingRowColors(True)
        self.ptt.setHorizontalHeaderLabels(["ID", "Balandlik (m)", "Tur"])
        self.ptt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ptt.verticalHeader().hide()
        self.ptt.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        l.addWidget(self.ptt)
        for _ in range(3):
            self.apr()
        self.ptt.cellWidget(0, 2).setCurrentIndex(1)

    def build_otab(self):
        l = QVBoxLayout(self.ot)
        l.setContentsMargins(12, 12, 12, 12)
        info = QLabel("Nuqtalar orasidagi balandlik farqlarini va o'rtacha xatolarni kiriting.")
        info.setStyleSheet("color: #8b949e; font-size: 13px; padding: 4px 0;")
        l.addWidget(info)
        br = QHBoxLayout()
        ab = QPushButton("+ Qo'shish")
        ab.clicked.connect(self.aor)
        br.addWidget(ab)
        br.addStretch()
        l.addLayout(br)
        self.ott = QTableWidget(0, 4)
        self.ott.setAlternatingRowColors(True)
        self.ott.setHorizontalHeaderLabels(["From", "To", "Farq (m)", "O'rt.xato (mm)"])
        self.ott.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ott.verticalHeader().hide()
        self.ott.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        l.addWidget(self.ott)
        for _ in range(3):
            self.aor()

    def sp(self, mn=0, mx=999999, v=0, d=0):
        s = QDoubleSpinBox() if d > 0 else QSpinBox()
        s.setMinimum(int(mn) if d == 0 else mn)
        s.setMaximum(int(mx) if d == 0 else mx)
        if d > 0:
            s.setDecimals(d)
            s.setValue(float(v))
        else:
            s.setValue(int(v))
        return s

    def apr(self):
        r = self.ptt.rowCount()
        self.ptt.insertRow(r)
        self.ptt.setRowHeight(r, 42)
        s = self.sp(1, 999999, r + 1)
        self.ptt.setCellWidget(r, 0, s)
        s2 = self.sp(-99999, 99999, 0, 4)
        s2.setSpecialValueText("—")
        s2.setEnabled(False)
        self.ptt.setCellWidget(r, 1, s2)
        cb = QComboBox()
        cb.addItems(["Sozlanadigan", "Fiksatsiyalangan"])
        cb.currentIndexChanged.connect(lambda i, rr=r: self.ptt.cellWidget(rr, 1).setEnabled(i == 1))
        self.ptt.setCellWidget(r, 2, cb)

    def aor(self):
        r = self.ott.rowCount()
        self.ott.insertRow(r)
        self.ott.setRowHeight(r, 42)
        for c in range(4):
            if c < 2:
                self.ott.setCellWidget(r, c, self.sp(1, 999999, 1 if c == 0 else 2))
            elif c == 2:
                self.ott.setCellWidget(r, c, self.sp(-9999, 9999, 0, 4))
            else:
                self.ott.setCellWidget(r, c, self.sp(0.01, 999, 2, 2))

    def pts(self):
        pp = []
        for r in range(self.ptt.rowCount()):
            iw = self.ptt.cellWidget(r, 0)
            hw = self.ptt.cellWidget(r, 1)
            tw = self.ptt.cellWidget(r, 2)
            if not all([iw, tw]):
                continue
            fix = tw.currentText() == "Fiksatsiyalangan"
            pp.append({"id": iw.value(), "height": hw.value() if fix and hw else None, "type": "fixed" if fix else "adjusted"})
        return pp

    def obs(self):
        oo = []
        for r in range(self.ott.rowCount()):
            w = [self.ott.cellWidget(r, c) for c in range(4)]
            if not all(w):
                continue
            oo.append({"from": w[0].value(), "to": w[1].value(), "val": w[2].value(), "stdev": w[3].value()})
        return oo

    def fmt_result(self, res, pts, obs):
        L = []
        L.append("╔" + "═" * 78 + "╗")
        L.append("║" + " " * 27 + "REPER KALIBRASH NATIJALARI" + " " * 27 + "║")
        L.append("╚" + "═" * 78 + "╝")
        L.append("")
        L.append(f"  Fiksatsiyalangan:  {sum(1 for p in pts if p['type'] == 'fixed')} ta")
        L.append(f"  Sozlanadigan:      {res['n_unknowns']} ta")
        L.append(f"  O'lchovlar:        {res['n_obs']} ta")
        L.append(f"  Erkinlik:          {res['dof']}")
        L.append(f"  Sigma0:            {res['s0']:.6f}" + ("  (yaxshi)" if res['s0'] < 2 else "  (tekshiring)"))
        L.append("")
        L.append("─" * 80)
        L.append("  SOZLANGAN BALANDLIKLAR")
        L.append("─" * 80)
        L.append(f"  {'ID':>6} {'Balandlik (m)':>18} {'Xato (mm)':>14}  {'Ishonch oralig\'i':>20}")
        L.append("  " + "─" * 62)
        for p in pts:
            if p["type"] == "fixed" and p["height"] is not None:
                L.append(f"  {p['id']:>6} {p['height']:>18.4f} {'—':>14}  {'(fiksatsiyalangan)':>20}")
        for pid in sorted(res["adjusted"]):
            h = res["adjusted"][pid]
            sd = res["stds"][pid] * 1000
            ci = sd * 1.96
            L.append(f"  {pid:>6} {h:>18.4f} {sd:>13.2f}  ±{ci:>14.2f} mm (95%)")
        L.append("")
        L.append("─" * 80)
        L.append("  QOLDIQLAR VA SINOV")
        L.append("─" * 80)
        L.append(f"  {'#':>3} {'From':>5} {'To':>5} {'O\'lchov(m)':>14} {'Qoldiq(mm)':>14}  {'w-stat':>8}  {'Holat':>10}")
        L.append("  " + "─" * 62)
        thr = 3.29
        for i, o in enumerate(obs):
            rv = res["residuals"][i] * 1000
            wv = res["w_obs"][i]
            flag = "✗ QO'POL" if wv > thr else "✓"
            L.append(f"  {i+1:>3} {o['from']:>5} {o['to']:>5} {o['val']:>14.4f} {rv:>13.2f}  {wv:>8.2f}  {flag:>10}")
        L.append("")
        if any(w > thr for w in res["w_obs"]):
            L.append("  ⚠ Qo'pol xatolar aniqlandi! (w > 3.29)")
        else:
            L.append("  ✓ Qo'pol xatolar aniqlanmadi.")
        L.append("")
        L.append("─" * 80)
        L.append("  MATRITSALAR")
        L.append("─" * 80)
        L.append(f"  Normal matritsa N = A'PA:")
        for r in range(res["Qxx"].shape[0]):
            row = "    "
            for c in range(res["Qxx"].shape[1]):
                row += f"{res['Qxx'][r, c]:>14.6f}"
            L.append(row)
        L.append("")
        L.append("─" * 80)
        L.append(f"  Reper Kalibrlash v2.0  |  Eng kichik kvadratlar usuli")
        L.append("─" * 80)
        return "\n".join(L)

    def gen_xml(self, pts, obs):
        L = []
        L.append('<?xml version="1.0" ?>')
        L.append('<gama-local xmlns="http://www.gnu.org/software/gama/gama-local">')
        L.append('<network>')
        L.append('<description>Reper kalibrlash</description>')
        L.append('<parameters sigma-act="aposteriori" conf-pr="0.95" />')
        L.append('<points-observations>')
        for p in pts:
            if p["type"] == "fixed" and p["height"] is not None:
                L.append(f'<point id="{p["id"]}" z="{p["height"]}" fix="z" />')
            else:
                L.append(f'<point id="{p["id"]}" adj="Z" />')
        for o in obs:
            L.append(f'<height-diff from="{o["from"]}" to="{o["to"]}" val="{o["val"]}" stdev="{o["stdev"]}" />')
        L.append('</points-observations>')
        L.append('</network>')
        L.append('</gama-local>')
        return "\n".join(L)

    def calc(self):
        pp = self.pts()
        oo = self.obs()
        if not pp:
            QMessageBox.warning(self, "Xatolik", "Nuqtalar kiriting!")
            return
        if not oo:
            QMessageBox.warning(self, "Xatolik", "O'lchovlar kiriting!")
            return
        if sum(1 for p in pp if p["type"] == "fixed") == 0:
            QMessageBox.warning(self, "Xatolik", "Kamida 1 ta fiksatsiyalangan nuqta kerak!")
            return
        if sum(1 for p in pp if p["type"] == "adjusted") == 0:
            QMessageBox.warning(self, "Xatolik", "Kamida 1 ta sozlanadigan nuqta kerak!")
            return
        self.pr.show()
        self.rb.setEnabled(False)
        self.out.setText("Hisoblanmoqda...")
        self.worker = Worker(pp, oo)
        self.worker.done.connect(self.on_done)
        self.worker.fail.connect(self.on_fail)
        self.worker.start()

    def on_done(self, res):
        self.pr.hide()
        self.rb.setEnabled(True)
        self.last = res
        pp = self.pts()
        oo = self.obs()
        self.out.setPlainText(self.fmt_result(res, pp, oo))

    def on_fail(self, msg):
        self.pr.hide()
        self.rb.setEnabled(True)
        self.out.setText(f"XATO: {msg}")
        QMessageBox.critical(self, "Xatolik", msg)

    def export_xml(self):
        if not self.last:
            QMessageBox.warning(self, "Ogohlantirish", "Avval hisoblashni bajarib oling!")
            return
        pp = self.pts()
        oo = self.obs()
        xml = self.gen_xml(pp, oo)
        p, _ = QFileDialog.getSaveFileName(self, "XML saqlash", "kalibrlash.xml", "XML (*.xml)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(xml)

    def export_txt(self):
        if not self.last:
            QMessageBox.warning(self, "Ogohlantirish", "Avval hisoblashni bajarib oling!")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Natijani saqlash", "natija.txt", "Matn (*.txt)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.out.toPlainText())

    def clear(self):
        r = QMessageBox.question(self, "Tozalash", "Barcha ma'lumotlarni tozalaysizmi?",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes:
            while self.ptt.rowCount(): self.ptt.removeRow(0)
            while self.ott.rowCount(): self.ott.removeRow(0)
            for _ in range(3): self.apr(); self.aor()
            self.ptt.cellWidget(0, 2).setCurrentIndex(1)
            self.out.clear()
            self.last = None

    def example(self):
        while self.ptt.rowCount(): self.ptt.removeRow(0)
        while self.ott.rowCount(): self.ott.removeRow(0)
        for pid, h, t in [(1, 100.0000, "fixed"), (2, None, "adjusted"), (3, None, "adjusted"), (4, None, "adjusted"), (5, None, "adjusted")]:
            self.apr()
            r = self.ptt.rowCount() - 1
            self.ptt.cellWidget(r, 0).setValue(pid)
            if t == "fixed":
                self.ptt.cellWidget(r, 2).setCurrentIndex(1)
                self.ptt.cellWidget(r, 1).setValue(h)
        for f, t, v, s in [(1, 2, 5.432, 2.0), (2, 3, -2.100, 2.0), (1, 3, 3.332, 2.0), (2, 4, 3.150, 2.0), (3, 4, 5.250, 2.0), (1, 4, 8.582, 2.0), (2, 5, 7.800, 2.0), (4, 5, 4.648, 2.0), (3, 5, 9.898, 2.0)]:
            self.aor()
            r = self.ott.rowCount() - 1
            self.ott.cellWidget(r, 0).setValue(f)
            self.ott.cellWidget(r, 1).setValue(t)
            self.ott.cellWidget(r, 2).setValue(v)
            self.ott.cellWidget(r, 3).setValue(s)
        self.out.setText("Namuna yuklandi. Hisoblash tugmasini bosing.")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
