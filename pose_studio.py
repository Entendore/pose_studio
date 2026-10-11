"""
PySide6 Robot Visualizer POSE STUDIO — MuJoCo Physics (Iteration 10)
==================================================================
A professional robotics simulator with Advanced Kinematic Interactions, 
Multi-Viewport CAD Layout, Collision Inspector, and ZMP analysis.
"""

import sys
import math
import datetime
import numpy as np
import os
import json
import socket
import threading
import queue
import random
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton,
    QSlider, QLabel, QToolBar, QStatusBar, QGroupBox,
    QFormLayout, QCheckBox, QAction, QPlainTextEdit, QMenu,
    QFileDialog, QTabWidget, QSplitter, QScrollArea, QFrame,
    QListWidget, QInputDialog, QComboBox, QLineEdit, QMessageBox,
    QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QRect, QRectF, QPointF, QSettings
from PySide6.QtGui import (
    QImage, QPainter, QColor, QFont, QMouseEvent, QWheelEvent, QPen, QBrush,
    QPainterPath, QLinearGradient, QRadialGradient
)
import mujoco

# ============================================================================
# TCP API Server Thread
# ============================================================================
class APIServerThread(threading.Thread):
    def __init__(self, port, cmd_queue, state_queue):
        super().__init__(daemon=True)
        self.port = port
        self.cmd_queue = cmd_queue
        self.state_queue = state_queue
        self.running = False
        self.sock = None

    def run(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("127.0.0.1", self.port))
            self.sock.listen(1)
            self.sock.settimeout(1.0)
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    conn.settimeout(1.0)
                    while self.running:
                        try:
                            data = conn.recv(4096)
                            if not data: break
                            cmd = json.loads(data.decode('utf-8'))
                            self.cmd_queue.put(cmd)
                            
                            if cmd.get("cmd") == "get_state":
                                if not self.state_queue.empty():
                                    state = self.state_queue.get()
                                    conn.sendall(json.dumps(state).encode('utf-8'))
                        except socket.timeout:
                            continue
                        except json.JSONDecodeError:
                            continue
                    conn.close()
                except socket.timeout:
                    continue
        except Exception as e:
            print(f"API Server Error: {e}")
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False


# ============================================================================
# Custom Widgets: Plots, Graphs & LiDAR
# ============================================================================
class IMUGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.max_hist = 200
        self.hist_x = deque(maxlen=self.max_hist)
        self.hist_y = deque(maxlen=self.max_hist)
        self.hist_z = deque(maxlen=self.max_hist)

    def push_data(self, gx, gy, gz=0.0):
        self.hist_x.append(gx); self.hist_y.append(gy); self.hist_z.append(gz)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 15, 15))
        p.setPen(QPen(QColor(35, 35, 35), 1, Qt.DashLine))
        for i in range(1, 4):
            p.drawLine(0, self.height() * i // 4, self.width(), self.height() * i // 4)
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        p.setPen(QPen(QColor(50, 200, 50), 2)); self._draw_line(p, self.hist_x)
        p.setPen(QPen(QColor(200, 50, 50), 2)); self._draw_line(p, self.hist_y)
        p.setPen(QPen(QColor(50, 200, 200), 2)); self._draw_line(p, self.hist_z)

        p.setPen(QColor(120, 120, 120)); p.setFont(QFont("Consolas", 8))
        p.drawText(5, 12, "IMU Gyro (X=Grn, Y=Red, Z=Cyn) rad/s")

    def _draw_line(self, p, data):
        if len(data) < 2: return
        step = self.width() / self.max_hist
        path = QPainterPath(QPointF(0, self.height() / 2 - (data[0] / 5.0) * (self.height() / 2)))
        for i, val in enumerate(list(data)[1:], 1):
            path.lineTo(i * step, self.height() / 2 - (val / 5.0) * (self.height() / 2))
        p.drawPath(path)


class TorquePlot(QWidget):
    def __init__(self, num_actuators, parent=None):
        super().__init__(parent); self.setMinimumHeight(150); self.n = num_actuators
        self.max_hist = 200
        self.hist = [deque(maxlen=self.max_hist) for _ in range(self.n)]
        self.colors = [QColor(255, 100, 100), QColor(100, 255, 100), QColor(100, 100, 255),
                       QColor(255, 255, 100), QColor(255, 100, 255), QColor(100, 255, 255),
                       QColor(200, 200, 200), QColor(150, 150, 250), QColor(250, 150, 150),
                       QColor(150, 250, 150), QColor(180, 180, 100), QColor(180, 100, 180),
                       QColor(100, 180, 180), QColor(220, 220, 220), QColor(255, 0, 0), QColor(0, 255, 0)]

    def push_data(self, torques):
        for i in range(self.n):
            self.hist[i].append(float(torques[i]))
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 15, 15))
        p.setPen(QPen(QColor(35, 35, 35), 1, Qt.DashLine))
        for i in range(1, 4):
            p.drawLine(0, self.height() * i // 4, self.width(), self.height() * i // 4)
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawLine(0, self.height() // 2, self.width(), self.height() // 2)
        p.setPen(QColor(120, 120, 120)); p.setFont(QFont("Consolas", 8))
        p.drawText(5, 12, "Actuator Torques (Nm) [-40, 40]")
        for i in range(self.n):
            if len(self.hist[i]) < 2: continue
            p.setPen(QPen(self.colors[i % len(self.colors)], 1.5))
            step = self.width() / self.max_hist
            path = QPainterPath(QPointF(0, self.height() / 2 - (self.hist[i][0] / 40.0) * (self.height() / 2)))
            for j, val in enumerate(list(self.hist[i])[1:], 1):
                path.lineTo(j * step, self.height() / 2 - (val / 40.0) * (self.height() / 2))
            p.drawPath(path)


class LidarPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(300)
        self.ranges = None; self.max_range = 6.0
        self.sweep_angle = 0.0

    def update_scan(self, ranges):
        self.ranges = ranges
        self.sweep_angle = (self.sweep_angle + 0.15) % (2 * math.pi)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))
        cx = self.width() // 2; cy = self.height() // 2
        max_r = min(cx, cy) - 25; scale = max_r / self.max_range

        p.setPen(QPen(QColor(40, 40, 40), 1, Qt.DashLine))
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            p.drawEllipse(QPointF(cx, cy), int(r * scale), int(r * scale))
            p.drawText(cx + int(r * scale) + 2, cy - 2, f"{r:.0f}m")

        p.setPen(QColor(60, 60, 60))
        p.drawLine(cx - max_r, cy, cx + max_r, cy)
        p.drawLine(cx, cy - max_r, cx, cy + max_r)
        p.setPen(QColor(120, 120, 120)); p.setFont(QFont("Consolas", 8))
        p.drawText(cx - 15, cy - max_r - 5, "Fwd (0°)")
        p.drawText(cx - 20, cy + max_r + 15, "Back (180°)")
        p.drawText(cx + max_r + 5, cy + 4, "90° (R)")
        p.drawText(cx - max_r - 40, cy + 4, "270° (L)")

        if self.ranges is not None:
            n = len(self.ranges)
            for i in range(n):
                angle = (i / n) * 2 * math.pi
                r = self.ranges[i]
                if r >= self.max_range: continue
                x = cx + r * math.sin(angle) * scale
                y = cy - r * math.cos(angle) * scale
                ratio = r / self.max_range
                color = QColor(int(255 * (1 - ratio)), int(255 * ratio), 0)
                p.setPen(QPen(color, 1, Qt.SolidLine))
                p.drawLine(cx, cy, int(x), int(y))
                p.setPen(QPen(color, 5)); p.drawPoint(int(x), int(y))

            sweep_x = cx + max_r * math.sin(self.sweep_angle)
            sweep_y = cy - max_r * math.cos(self.sweep_angle)
            grad = QLinearGradient(cx, cy, sweep_x, sweep_y)
            grad.setColorAt(0, QColor(0, 255, 150, 100))
            grad.setColorAt(1, QColor(0, 255, 150, 0))
            p.setBrush(QBrush(grad)); p.setPen(Qt.NoPen)
            span_angle = 20.0
            start_angle_deg = 90.0 - math.degrees(self.sweep_angle) - span_angle
            p.drawPie(QRectF(cx - max_r, cy - max_r, max_r * 2, max_r * 2),
                      int(start_angle_deg * 16), int(span_angle * 16))

        p.setPen(QColor(200, 200, 200)); p.setFont(QFont("Consolas", 9))
        p.drawText(10, 15, "LIDAR SCAN (120 Rays)")


class CoMPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(150)
        self.hist_x = deque(maxlen=500)
        self.hist_y = deque(maxlen=500)

    def push_data(self, x, y):
        self.hist_x.append(x); self.hist_y.append(y)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))
        cx = self.width() // 2; cy = self.height() // 2
        scale = 20.0  # 1m = 20px

        p.setPen(QPen(QColor(40, 40, 40), 1, Qt.DashLine))
        p.drawLine(cx, 0, cx, self.height())
        p.drawLine(0, cy, self.width(), cy)

        if len(self.hist_x) > 1:
            p.setPen(QPen(QColor(0, 255, 150), 2))
            path = QPainterPath(QPointF(cx + self.hist_x[0] * scale, cy - self.hist_y[0] * scale))
            for i in range(1, len(self.hist_x)):
                path.lineTo(cx + self.hist_x[i] * scale, cy - self.hist_y[i] * scale)
            p.drawPath(path)
            
            p.setBrush(QBrush(QColor(255, 0, 0)))
            p.drawEllipse(QPointF(cx + self.hist_x[-1] * scale, cy - self.hist_y[-1] * scale), 4, 4)

        p.setPen(QColor(120, 120, 120)); p.setFont(QFont("Consolas", 8))
        p.drawText(5, 12, "Center of Mass Trajectory (Top-Down)")


# ============================================================================
# 3D Viewport
# ============================================================================
class MuJoCoViewport(QWidget):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model = model; self.data = data; self.parent_win = parent
        self.setMinimumSize(640, 480); self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._renderer = None; self._pip_renderer = None
        self._img_w = 320; self._img_h = 240; self._pip_w = 240; self._pip_h = 180
        self._render_failed = False

        self.view_layout = 1  # 1, 2, or 4
        
        self.cam_main = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam_main)
        self.cam_main.distance = 3.5; self.cam_main.elevation = -25; self.cam_main.azimuth = 135
        self.cam_main.lookat[:] = [0.0, 0.0, 0.25]
        
        self.cam_top = mujoco.MjvCamera()
        self.cam_top.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_top.fixedcamid = -1  # Free look
        self.cam_top.distance = 3.0; self.cam_top.elevation = -90; self.cam_top.azimuth = 0
        self.cam_top.lookat[:] = [0.0, 0.0, 0.0]

        self.cam_side = mujoco.MjvCamera()
        self.cam_side.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_side.fixedcamid = -1
        self.cam_side.distance = 3.0; self.cam_side.elevation = 0; self.cam_side.azimuth = 0
        self.cam_side.lookat[:] = [0.0, 0.0, 0.0]

        self.cam_front = mujoco.MjvCamera()
        self.cam_front.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_front.fixedcamid = -1
        self.cam_front.distance = 3.0; self.cam_front.elevation = 0; self.cam_front.azimuth = 90
        self.cam_front.lookat[:] = [0.0, 0.0, 0.0]

        self.opt = mujoco.MjvOption()
        self.opt.label = mujoco.mjtLabel.mjLABEL_NONE
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        self.pip_opt = mujoco.MjvOption()

        self._last_pos = None; self._button = None
        self._image = None; self._image2 = None; self._image3 = None; self._image4 = None
        self._pip_image = None; self._scaled_geom = None
        self._active_cam = self.cam_main
        self._active_rect = QRect()

        self.nav_goal_mode = False
        self.dragging_trunk = False
        self.dragging_trunk_rotate = False
        self.dragging_chain = []
        self.dragging_ik_body = -1
        self.drag_last_pos = None
        self._drag_start_qpos = None

        self._cache_ids()

    def _cache_ids(self):
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.vel_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_vel")
        self.h_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "height_sensor")
        self.gyro_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.foot_sensors = ['fl_touch', 'fr_touch', 'hl_touch', 'hr_touch']
        self.foot_sids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                          for n in self.foot_sensors]
                          
        self.foot_bodies = ['fl_calf', 'fr_calf', 'hl_calf', 'hr_calf']
        self.foot_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.foot_bodies]
        self.gripper_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

    def update_model(self, model, data):
        self.model = model; self.data = data
        if self._renderer: self._renderer.close()
        if self._pip_renderer: self._pip_renderer.close()
        self._renderer = None; self._pip_renderer = None
        self._ensure_renderer(); self._cache_ids()

    def _ensure_renderer(self):
        w = max(self.width(), 320); h = max(self.height(), 240)
        if self._renderer is None or w != self._img_w or h != self._img_h:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._img_w, self._img_h = w, h
        if self._pip_renderer is None:
            self._pip_renderer = mujoco.Renderer(self.model, height=self._pip_h, width=self._pip_w)

    def resizeEvent(self, ev): self._ensure_renderer(); super().resizeEvent(ev)
    def showEvent(self, ev): self._ensure_renderer(); super().showEvent(ev)

    def render_scene(self):
        if self._renderer is None: self._ensure_renderer()
        if self._renderer is None: return
        
        self._render_failed = False
        
        # Update Fixed Cams lookat
        if self.trunk_id >= 0:
            pos = self.data.xpos[self.trunk_id]
            self.cam_top.lookat[:] = [pos[0], pos[1], 0]
            self.cam_side.lookat[:] = [pos[0], pos[1], 0.3]
            self.cam_front.lookat[:] = [pos[0], pos[1], 0.3]
        
        try:
            self._renderer.update_scene(self.data, self.cam_main, scene_option=self.opt)
            self._draw_custom_geoms(self._renderer.scn)
            self._renderer.scn.flags |= mujoco.mjtRndFlag.mjRND_SHADOW
            pixels = self._renderer.render()
            h, w, _ = pixels.shape
            self._image = QImage(pixels.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
            
            if self.view_layout >= 2:
                self._renderer.update_scene(self.data, self.cam_top, scene_option=self.opt)
                self._draw_custom_geoms(self._renderer.scn)
                pixels2 = self._renderer.render()
                h2, w2, _ = pixels2.shape
                self._image2 = QImage(pixels2.tobytes(), w2, h2, w2 * 3, QImage.Format_RGB888).copy()
                
            if self.view_layout >= 4:
                self._renderer.update_scene(self.data, self.cam_side, scene_option=self.opt)
                self._draw_custom_geoms(self._renderer.scn)
                pixels3 = self._renderer.render()
                h3, w3, _ = pixels3.shape
                self._image3 = QImage(pixels3.tobytes(), w3, h3, w3 * 3, QImage.Format_RGB888).copy()
                
                self._renderer.update_scene(self.data, self.cam_front, scene_option=self.opt)
                self._draw_custom_geoms(self._renderer.scn)
                pixels4 = self._renderer.render()
                h4, w4, _ = pixels4.shape
                self._image4 = QImage(pixels4.tobytes(), w4, h4, w4 * 3, QImage.Format_RGB888).copy()

            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam")
            if cam_id >= 0 and self._pip_renderer:
                self._pip_renderer.update_scene(self.data, cam_id, scene_option=self.pip_opt)
                pip_px = self._pip_renderer.render()
                ph, pw, _ = pip_px.shape
                self._pip_image = QImage(pip_px.tobytes(), pw, ph, pw * 3, QImage.Format_RGB888).copy()
        except Exception as e:
            if self.parent_win and not self._render_failed:
                self.parent_win.log(f"[ERROR] Render: {e}")
                self._render_failed = True
        self.update()

    def _draw_custom_geoms(self, scn):
        if self.parent_win.wind_x != 0 or self.parent_win.wind_y != 0:
            pos = self.data.xpos[self.trunk_id].copy()
            vec = np.array([self.parent_win.wind_x, self.parent_win.wind_y, 0])
            self._add_arrow_geom(scn, pos, vec, QColor(0, 150, 255, 200))
            
        if self.vel_sid >= 0:
            adr = self.model.sensor_adr[self.vel_sid]
            vx = self.data.sensordata[adr]; vy = self.data.sensordata[adr+1]
            speed = math.hypot(vx, vy)
            if speed > 0.1:
                pos = self.data.xpos[self.trunk_id].copy()
                vec = np.array([vx, vy, 0]) * 0.5
                self._add_arrow_geom(scn, pos, vec, QColor(0, 255, 100, 200))

        if self.parent_win.path_pts:
            for i in range(len(self.parent_win.path_pts) - 1):
                p1 = np.array(self.parent_win.path_pts[i])
                p2 = np.array(self.parent_win.path_pts[i+1])
                self._add_line_geom(scn, p1, p2, QColor(255, 165, 0, 150))
            for pt in self.parent_win.waypoints:
                self._add_sphere_geom(scn, np.array(pt), 0.05, QColor(255, 165, 0, 200))

        if self.parent_win.cop_valid:
            cop_pos = np.array([self.parent_win.cop_xy[0], self.parent_win.cop_xy[1], 0.02])
            color = QColor(0, 255, 0, 255) if self.parent_win.zmp_inside else QColor(255, 0, 0, 255)
            self._add_sphere_geom(scn, cop_pos, 0.04, color)
            pts = self.parent_win.support_poly_pts
            for i in range(len(pts)):
                p1 = np.array([pts[i][0], pts[i][1], 0.01])
                p2 = np.array([pts[(i+1)%len(pts)][0], pts[(i+1)%len(pts)][1], 0.01])
                self._add_line_geom(scn, p1, p2, QColor(0, 255, 255, 100))

        # Collision Inspector Highlights
        if self.parent_win.selected_coll_geoms:
            for gid in self.parent_win.selected_coll_geoms:
                if gid >= 0:
                    pos = self.data.geom_xpos[gid]
                    self._add_sphere_geom(scn, pos, 0.08, QColor(255, 0, 255, 100))

    def _add_arrow_geom(self, scn, pos, vec, color):
        if scn.ngeom >= scn.maxgeom: return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.02, 0.02, np.linalg.norm(vec)]),
            pos=pos + vec/2,
            mat=np.eye(3).flatten(),
            rgba=np.array([color.redF(), color.greenF(), color.blueF(), color.alphaF()])
        )
        v = vec / (np.linalg.norm(vec) + 1e-6)
        z = np.array([0, 0, 1])
        axis = np.cross(z, v)
        angle = np.arccos(np.dot(z, v))
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            mujoco.mju_quat2Mat(scn.geoms[scn.ngeom].mat, np.array([math.cos(angle/2), *(axis*math.sin(angle/2))]))
        scn.ngeom += 1

    def _add_sphere_geom(self, scn, pos, r, color):
        if scn.ngeom >= scn.maxgeom: return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([r, r, r]),
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([color.redF(), color.greenF(), color.blueF(), color.alphaF()])
        )
        scn.ngeom += 1

    def _add_line_geom(self, scn, p1, p2, color):
        if scn.ngeom >= scn.maxgeom: return
        mid = (p1 + p2) / 2.0
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: return
        
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.01, 0.01, length/2]),
            pos=mid,
            mat=np.eye(3).flatten(),
            rgba=np.array([color.redF(), color.greenF(), color.blueF(), color.alphaF()])
        )
        v = vec / length
        z = np.array([0, 0, 1])
        axis = np.cross(z, v)
        angle = np.arccos(np.dot(z, v))
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            mujoco.mju_quat2Mat(scn.geoms[scn.ngeom].mat, np.array([math.cos(angle/2), *(axis*math.sin(angle/2))]))
        scn.ngeom += 1

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(25, 25, 30))
        
        rects = self._get_view_rects()
        
        if self._image is not None and not self._image.isNull():
            p.drawImage(rects[0], self._image.scaled(rects[0].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        if self.view_layout >= 2 and self._image2 is not None:
            p.drawImage(rects[1], self._image2.scaled(rects[1].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        if self.view_layout >= 4 and self._image3 is not None and self._image4 is not None:
            p.drawImage(rects[2], self._image3.scaled(rects[2].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            p.drawImage(rects[3], self._image4.scaled(rects[3].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Draw borders
        p.setPen(QPen(QColor(0, 0, 0), 2))
        for r in rects:
            p.drawRect(r)
            
        # Labels
        p.setPen(QColor(200, 200, 200, 150)); p.setFont(QFont("Consolas", 8))
        labels = ["Perspective", "Top", "Side", "Front"]
        for i, r in enumerate(rects):
            p.drawText(r.left()+5, r.top()+12, labels[i])

        if self._pip_image is not None and not self._pip_image.isNull():
            px = self.width() - self._pip_w - 20; py = self.height() - self._pip_h - 20
            p.fillRect(px - 3, py - 3, self._pip_w + 6, self._pip_h + 6, QColor(0, 0, 0, 180))
            p.drawImage(px, py, self._pip_image)
            p.setPen(QColor(0, 255, 150, 200))
            p.drawRoundedRect(px - 3, py - 3, self._pip_w + 6, self._pip_h + 6, 4, 4)
            p.setPen(QColor(255, 255, 255, 150)); p.setFont(QFont("Consolas", 8))
            p.drawText(px + 5, py + 14, f"ROBOT CAM ({self.parent_win.pip_mode})")

        self._draw_hud(p)
        self._draw_attitude_indicator(p)

        p.setPen(QColor(200, 200, 200, 150)); p.setFont(QFont("Consolas", 8))
        if self.parent_win.kinematic_mode:
            mode_txt = "SKELETON: Drag Trunk=Move, Alt+Drag=Rotate, Drag Feet=IK | Ctrl+Z: Undo"
        else:
            mode_txt = "RMB: Menu | Ctrl+LMB: Select | Shift+LMB: Waypoints | WASD/QE: Drive | G: Grab"
        p.drawText(10, self.height() - 10, mode_txt)

    def _get_view_rects(self):
        w = self.width(); h = self.height()
        if self.view_layout == 1:
            return [QRect(0, 0, w, h)]
        elif self.view_layout == 2:
            return [QRect(0, 0, w//2, h), QRect(w//2, 0, w-w//2, h)]
        else:
            return [QRect(0, 0, w//2, h//2), QRect(w//2, 0, w-w//2, h//2),
                    QRect(0, h//2, w//2, h-h//2), QRect(w//2, h//2, w-w//2, h-h//2)]

    def _draw_hud(self, p: QPainter):
        hud_rect = QRect(10, 10, 230, 180)
        p.setBrush(QBrush(QColor(15, 15, 20, 200))); p.setPen(Qt.NoPen)
        p.drawRoundedRect(hud_rect, 6, 6)

        p.setPen(QColor(0, 255, 150) if not self.parent_win.kinematic_mode else QColor(255, 165, 0))
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        mode_title = "KINEMATIC EDITOR" if self.parent_win.kinematic_mode else "ROBOT TELEMETRY"
        p.drawText(hud_rect.adjusted(12, 8, 0, 0), Qt.AlignTop, mode_title)

        p.setPen(QColor(60, 60, 60)); p.drawLine(20, 32, 230, 32)
        p.setPen(QColor(220, 220, 220)); p.setFont(QFont("Consolas", 9))

        speed = 0.0
        if self.vel_sid >= 0:
            adr = self.model.sensor_adr[self.vel_sid]
            vx = self.data.sensordata[adr]; vy = self.data.sensordata[adr + 1]
            speed = math.hypot(vx, vy)

        height_val = 0.0
        if self.h_sid >= 0:
            adr = self.model.sensor_adr[self.h_sid]
            dist = self.data.sensordata[adr]
            if dist > 0: height_val = dist

        ke = self.data.energy[1] if len(self.data.energy) > 1 else 0.0
        pe = self.data.energy[0] if len(self.data.energy) > 0 else 0.0
        bat = self.parent_win.battery_percent

        txt = (
            f"Time:    {self.data.time:.2f} s\n"
            f"Speed:   {speed:.2f} m/s\n"
            f"Height:  {height_val:.2f} m\n"
            f"Energy:  K:{ke:.1f} P:{pe:.1f}\n"
            f"Battery: {bat:.0f}%\n"
            f"Grabbing: {'YES' if self.parent_win.is_grabbing else 'NO'}"
        )
        p.drawText(hud_rect.adjusted(12, 38, 0, 0), Qt.AlignTop, txt)
        
        if self.parent_win.nav_goal is not None:
            p.setPen(QColor(0, 255, 150))
            p.drawText(hud_rect.adjusted(12, 130, 0, 0), Qt.AlignTop, f"Nav: ({self.parent_win.nav_goal[0]:.1f}, {self.parent_win.nav_goal[1]:.1f})")
        elif self.parent_win.path_pts:
            p.setPen(QColor(255, 165, 0))
            p.drawText(hud_rect.adjusted(12, 130, 0, 0), Qt.AlignTop, f"Path: {len(self.parent_win.waypoints)} pts")
            
        p.setPen(QColor(0, 255, 150) if self.parent_win.zmp_inside else QColor(255, 50, 50))
        p.drawText(hud_rect.adjusted(12, 150, 0, 0), Qt.AlignTop, f"ZMP: {'STABLE' if self.parent_win.zmp_inside else 'FALLING!'}")

        nav_rect = QRect(10, 200, 230, 60)
        p.setBrush(QBrush(QColor(15, 15, 20, 200))); p.setPen(Qt.NoPen)
        p.drawRoundedRect(nav_rect, 6, 6)
        p.setPen(QColor(0, 150, 255)); p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.drawText(nav_rect.adjusted(12, 8, 0, 0), Qt.AlignTop, "NAVIGATION & CONTACTS")

        avoid_status = "BLOCKED" if self.parent_win.obstacle_blocked else "CLEAR"
        avoid_color = QColor(255, 80, 80) if self.parent_win.obstacle_blocked else QColor(80, 255, 80)
        p.setPen(QColor(220, 220, 220)); p.setFont(QFont("Consolas", 9))
        p.drawText(nav_rect.adjusted(12, 26, 0, 0), Qt.AlignTop, "LiDAR: ")
        p.setPen(avoid_color)
        p.drawText(nav_rect.adjusted(60, 26, 0, 0), Qt.AlignTop, avoid_status)

        p.setPen(QColor(220, 220, 220))
        p.drawText(nav_rect.adjusted(12, 44, 0, 0), Qt.AlignTop, "Feet:")
        fx_positions = [60, 100, 140, 180]
        foot_labels = ['FL', 'FR', 'HL', 'HR']
        for i, sid in enumerate(self.foot_sids):
            is_touch = False
            if sid >= 0:
                adr = self.model.sensor_adr[sid]
                if self.data.sensordata[adr] > 0.1: is_touch = True
            color = QColor(255, 50, 50) if is_touch else QColor(60, 60, 60)
            p.setBrush(QBrush(color)); p.setPen(Qt.NoPen)
            p.drawEllipse(nav_rect.left() + fx_positions[i], nav_rect.top() + 48, 8, 8)
            p.setPen(QColor(150, 150, 150)); p.setFont(QFont("Consolas", 7))
            p.drawText(nav_rect.left() + fx_positions[i] - 4, nav_rect.top() + 44, foot_labels[i])

    def _draw_attitude_indicator(self, p: QPainter):
        if self.trunk_id < 0: return
        quat = self.data.xquat[self.trunk_id]
        w, x, y, z = quat
        sinr = 2.0 * (w * x + y * z); cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x); pitch = math.asin(max(-1.0, min(1.0, sinp)))

        cx = self.width() - 80; cy = 80; r = 50
        p.setBrush(QBrush(QColor(15, 15, 20, 200))); p.setPen(QPen(QColor(60, 60, 60), 2))
        p.drawEllipse(cx, cy, r, r)

        p.save()
        p.translate(cx, cy)
        p.rotate(math.degrees(roll))
        pitch_offset = math.sin(pitch) * r * 1.5

        clip_path = QPainterPath()
        clip_path.addEllipse(-r, -r, r * 2, r * 2)
        p.setClipPath(clip_path)

        p.setBrush(QBrush(QColor(50, 100, 200))); p.setPen(Qt.NoPen)
        p.drawRect(-r, -r - pitch_offset, r * 2, r)
        p.setBrush(QBrush(QColor(100, 60, 30)))
        p.drawRect(-r, 0 - pitch_offset, r * 2, r)

        p.setPen(QPen(QColor(255, 255, 255, 120), 1))
        p.setFont(QFont("Consolas", 6))
        for deg in [-30, -20, -10, 10, 20, 30]:
            py = (deg / 30.0) * (r * 0.6) - pitch_offset
            if -r < py < r:
                length = r * 0.3 if deg % 10 == 0 else r * 0.15
                p.drawLine(-length, py, length, py)
                if deg % 10 == 0:
                    p.drawText(int(length + 2), int(py + 3), f"{deg}")

        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(-r, -pitch_offset, r, -pitch_offset)
        p.restore()

        p.setPen(QPen(QColor(255, 255, 0), 2))
        p.drawLine(cx - 10, cy, cx + 10, cy)
        p.drawLine(cx, cy - 10, cx, cy + 10)
        p.setBrush(QBrush(QColor(255, 255, 0)))
        p.drawEllipse(cx, cy, 4, 4)
        p.setPen(QColor(150, 150, 150)); p.setFont(QFont("Consolas", 8))
        p.drawText(QRect(cx - 50, cy + r + 5, 100, 15), Qt.AlignCenter, "ATTITUDE")
        p.drawText(QRect(cx - 60, cy + r + 20, 120, 15), Qt.AlignCenter, f"R:{math.degrees(roll):+.0f}° P:{math.degrees(pitch):+.0f}°")

    def mousePressEvent(self, ev: QMouseEvent):
        self._update_active_view(ev.position().toPoint())
        
        if self.nav_goal_mode and ev.button() == Qt.LeftButton:
            self._pick_nav_goal(ev.position().toPoint()); return
        if ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.ShiftModifier):
            self._pick_waypoint(ev.position().toPoint()); return
        if self.parent_win.kinematic_mode and ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.ControlModifier):
            self._drag_start_qpos = self.data.qpos.copy()
            self._pick_body_for_dragging(ev.position().toPoint(), bool(ev.modifiers() & Qt.AltModifier)); return
        if ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.ControlModifier):
            self._pick_body(ev.position().toPoint()); return
        if ev.button() == Qt.RightButton:
            if self.nav_goal_mode:
                self.nav_goal_mode = False
                self.setCursor(Qt.ArrowCursor)
                self.parent_win.log("[INFO] Navigation goal cancelled.")
            else:
                self._show_context_menu(ev.globalPosition().toPoint())
            return
        self._last_pos = ev.position().toPoint(); self._button = ev.button(); self.setFocus()

    def _update_active_view(self, pos):
        rects = self._get_view_rects()
        for i, r in enumerate(rects):
            if r.contains(pos):
                self._active_rect = r
                if i == 0: self._active_cam = self.cam_main
                elif i == 1: self._active_cam = self.cam_top
                elif i == 2: self._active_cam = self.cam_side
                elif i == 3: self._active_cam = self.cam_front
                break

    def _rel_coords(self, pos):
        if self._scaled_geom is None or self._renderer is None: return None
        r = self._active_rect
        local_x = pos.x() - r.left()
        local_y = pos.y() - r.top()
        rel_x = local_x / r.width()
        rel_y = local_y / r.height()
        if 0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0:
            return rel_x, 1.0 - rel_y
        return None

    def _pick_body_for_dragging(self, pos, is_alt_pressed):
        rc = self._rel_coords(pos)
        if rc is None: return
        rel_x, rel_y = rc
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32); skinid = np.array([-1], dtype=np.int32)
        aspect = self._img_w / self._img_h
        mujoco.mjv_select(self.model, self.data, self.opt, aspect, rel_x, rel_y,
                          self._renderer.scn, selpnt, geomid, skinid)
        if geomid[0] >= 0:
            bid = self.model.geom_bodyid[geomid[0]]
            if bid == self.trunk_id:
                self.dragging_trunk = True
                self.dragging_trunk_rotate = is_alt_pressed
                self.drag_last_pos = pos
                self.setCursor(Qt.SizeAllCursor)
                msg = "[SKELETON] Grabbed Trunk. Drag to rotate yaw." if is_alt_pressed else "[SKELETON] Grabbed Trunk. Drag to move."
                self.parent_win.log(msg)
            elif bid in self.foot_body_ids:
                self.dragging_ik_body = bid
                self.drag_last_pos = pos
                self.setCursor(Qt.SizeAllCursor)
                self.parent_win.log("[SKELETON] Grabbed Foot for IK Targeting.")
            else:
                self.dragging_chain = []
                curr_bid = bid
                while curr_bid > 0 and curr_bid != self.trunk_id:
                    jnt_adr = self.model.body_jntadr[curr_bid]
                    jnt_num = self.model.body_jntnum[curr_bid]
                    for j in range(jnt_num):
                        jid = jnt_adr + j
                        if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                            self.dragging_chain.append(jid)
                    curr_bid = int(self.model.body_parentid[curr_bid])
                if self.dragging_chain:
                    self.drag_last_pos = pos
                    self.setCursor(Qt.SizeAllCursor)
                    jnames = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                              for j in self.dragging_chain]
                    self.parent_win.log(f"[SKELETON] Grabbed FK Chain: {jnames}")

    def _pick_body(self, pos):
        rc = self._rel_coords(pos)
        if rc is None: return
        rel_x, rel_y = rc
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32); skinid = np.array([-1], dtype=np.int32)
        aspect = self._img_w / self._img_h
        mujoco.mjv_select(self.model, self.data, self.opt, aspect, rel_x, rel_y,
                          self._renderer.scn, selpnt, geomid, skinid)
        if geomid[0] >= 0:
            bid = self.model.geom_bodyid[geomid[0]]
            self.parent_win.select_body_in_tree(bid)

    def _pick_nav_goal(self, pos):
        rc = self._rel_coords(pos)
        if rc is None: return
        rel_x, rel_y = rc
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32); skinid = np.array([-1], dtype=np.int32)
        aspect = self._img_w / self._img_h
        mujoco.mjv_select(self.model, self.data, self.opt, aspect, rel_x, rel_y,
                          self._renderer.scn, selpnt, geomid, skinid)
        target_geoms = {"ground", "stair1", "stair2", "stair3"}
        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0]) if geomid[0] >= 0 else ""
        if gname in target_geoms:
            self.parent_win.set_nav_goal(selpnt[:2])
        self.nav_goal_mode = False
        self.setCursor(Qt.ArrowCursor)

    def _pick_waypoint(self, pos):
        rc = self._rel_coords(pos)
        if rc is None: return
        rel_x, rel_y = rc
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32); skinid = np.array([-1], dtype=np.int32)
        aspect = self._img_w / self._img_h
        mujoco.mjv_select(self.model, self.data, self.opt, aspect, rel_x, rel_y,
                          self._renderer.scn, selpnt, geomid, skinid)
        target_geoms = {"ground", "stair1", "stair2", "stair3"}
        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0]) if geomid[0] >= 0 else ""
        if gname in target_geoms:
            self.parent_win.add_waypoint(selpnt[:2])

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.drag_last_pos is not None:
            pos = ev.position().toPoint()
            dx = pos.x() - self.drag_last_pos.x()
            dy = pos.y() - self.drag_last_pos.y()
            self.drag_last_pos = pos
            
            if self.dragging_trunk:
                if self.dragging_trunk_rotate:
                    delta_yaw = dx * 0.02
                    dq = np.array([math.cos(delta_yaw/2), 0, 0, math.sin(delta_yaw/2)])
                    cq = self.data.qpos[3:7].copy()
                    w1, x1, y1, z1 = dq
                    w2, x2, y2, z2 = cq
                    self.data.qpos[3:7] = [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]
                else:
                    az = math.radians(self._active_cam.azimuth)
                    right_x = math.sin(az); right_y = -math.cos(az)
                    self.data.qpos[0] += dx * 0.01 * right_x
                    self.data.qpos[1] += dx * 0.01 * right_y
                    self.data.qpos[2] -= dy * 0.01
                    self.data.qpos[2] = max(0.05, self.data.qpos[2])
            elif self.dragging_ik_body >= 0:
                az = math.radians(self._active_cam.azimuth)
                el = math.radians(self._active_cam.elevation)
                right = np.array([math.sin(az), -math.cos(az), 0])
                fwd = np.array([math.cos(az)*math.cos(el), math.sin(az)*math.cos(el), -math.sin(el)])
                up = np.cross(fwd, right)
                
                scale = 0.01
                delta_3d = right * (dx * scale) + up * (-dy * scale)
                
                foot_pos = self.data.xpos[self.dragging_ik_body].copy()
                target_pos = foot_pos + delta_3d
                
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacp, None, foot_pos, self.dragging_ik_body)
                
                leg_joints = []
                curr_bid = self.dragging_ik_body
                while curr_bid > 0 and curr_bid != self.trunk_id:
                    jnt_adr = self.model.body_jntadr[curr_bid]
                    jnt_num = self.model.body_jntnum[curr_bid]
                    for j in range(jnt_num):
                        jid = jnt_adr + j
                        if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                            leg_joints.append(jid)
                    curr_bid = int(self.model.body_parentid[curr_bid])
                    
                leg_dofs = [self.model.jnt_dofadr[j] for j in leg_joints]
                jac_leg = jacp[:, leg_dofs]
                
                lam = 0.5
                jtj = jac_leg @ jac_leg.T + np.eye(3) * (lam ** 2)
                dq = jac_leg.T @ np.linalg.inv(jtj) @ delta_3d
                
                for i, jid in enumerate(leg_joints):
                    qpos_adr = self.model.jnt_qposadr[jid]
                    new_val = self.data.qpos[qpos_adr] + dq[i]
                    rng = self.model.jnt_range[jid]
                    if rng[0] != 0 or rng[1] != 0:
                        new_val = max(rng[0], min(rng[1], new_val))
                    self.data.qpos[qpos_adr] = new_val
                    
            elif self.dragging_chain:
                delta = (dx - dy) * 0.015
                for i, jid in enumerate(self.dragging_chain):
                    qpos_adr = self.model.jnt_qposadr[jid]
                    new_val = self.data.qpos[qpos_adr] + delta * (0.8 ** i)
                    rng = self.model.jnt_range[jid]
                    if rng[0] != 0 or rng[1] != 0:
                        new_val = max(rng[0], min(rng[1], new_val))
                    self.data.qpos[qpos_adr] = new_val
                    
            mujoco.mj_forward(self.model, self.data)
            self.parent_win.update_sliders_from_qpos()
            return
            
        if self._last_pos is None: return
        pos = ev.position().toPoint()
        dx = pos.x() - self._last_pos.x(); dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._button == Qt.LeftButton:
            self._active_cam.azimuth -= dx * 0.5
            self._active_cam.elevation = max(-89.0, min(89.0, self._active_cam.elevation - dy * 0.5))
        elif self._button == Qt.RightButton:
            self._active_cam.distance = max(0.1, min(50.0, self._active_cam.distance * (1 + dy * 0.01)))
        elif self._button == Qt.MiddleButton:
            s = 0.0015 * self._active_cam.distance
            self._active_cam.lookat[0] += dx * s; self._active_cam.lookat[1] -= dy * s

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if self.drag_last_pos is not None:
            if self.parent_win.kinematic_mode and self._drag_start_qpos is not None:
                self.parent_win.push_undo(self._drag_start_qpos)
            self.dragging_trunk = False; self.dragging_trunk_rotate = False
            self.dragging_chain = []; self.dragging_ik_body = -1; self.drag_last_pos = None
            self._drag_start_qpos = None
            self.setCursor(Qt.ArrowCursor)
            self.parent_win.log("[SKELETON] Released.")
        self._last_pos = None; self._button = None

    def wheelEvent(self, ev: QWheelEvent):
        d = ev.angleDelta().y()
        self._active_cam.distance = max(0.1, min(50.0, self._active_cam.distance * (1 - d * 0.001)))

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("🎯 Set Navigation Goal", self._enter_nav_mode)
        menu.addSeparator()
        menu.addAction("Clear Waypoints", self.parent_win.clear_waypoints)
        menu.addSeparator()
        menu.addAction("Reset Robot Only", self.parent_win.reset_robot_only)
        menu.addAction("Reset Camera (C)", self._reset_cam)
        menu.exec(pos)

    def _enter_nav_mode(self):
        self.nav_goal_mode = True
        self.setCursor(Qt.CrossCursor)
        self.parent_win.log("[INFO] Nav mode. Click ground/stairs to set target. RMB to cancel.")

    def _reset_cam(self):
        if self.parent_win.a_track.isChecked():
            self.parent_win.a_track.setChecked(False)
            self.parent_win._toggle_track(False)
        self.cam_main.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_main.distance = 3.5; self.cam_main.elevation = -25; self.cam_main.azimuth = 135
        self.cam_main.lookat[:] = [0.0, 0.0, 0.25]


# ============================================================================
# Main Window
# ============================================================================
class RobotSimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Robot Visualizer POSE STUDIO — MuJoCo")
        self.resize(1600, 900)

        self.current_env_xml = ENV_FLAT
        self.model = mujoco.MjModel.from_xml_string(self.current_env_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.act_qpos = np.zeros(self.model.nu, dtype=int)
        self.act_qvel = np.zeros(self.model.nu, dtype=int)
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            self.act_qpos[i] = self.model.jnt_qposadr[jid]
            self.act_qvel[i] = self.model.jnt_dofadr[jid]

        self._rebuild_caches()

        self.sim_time = 0.0; self.sim_running = False; self.sim_speed = 1.0
        self.gait_enabled = True; self.gait_freq = 1.5; self.current_gait = "Trot"
        self.manual_targets = np.zeros(self.model.nu)

        self.recording = False; self.rec_folder = ""; self.rec_frame = 0
        self.rec_target_fps = 30
        self._last_rec_t = 0.0

        self.target_vx = 0.0; self.target_vy = 0.0; self.target_wz = 0.0
        self.cmd_vx = 0.0; self.cmd_vy = 0.0; self.cmd_wz = 0.0

        self.nav_goal = None
        self.robot_yaw = 0.0
        self.obstacle_blocked = False
        self.recovering = False; self.recover_timer = 0.0
        self.battery_percent = 100.0
        self.chase_cam = False
        self._battery_dead_logged = False

        self.trail_mocap_ids = []
        for i in range(50):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trail_{i}")
            self.trail_mocap_ids.append(self.model.body_mocapid[bid])
        self.trail_idx = 0; self.trail_step = 0

        goal_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self.goal_mocap_id = self.model.body_mocapid[goal_bid] if goal_bid >= 0 else -1

        self.wind_x = 0.0; self.wind_y = 0.0
        self.saved_poses = {}

        self.lidar_ranges = np.zeros(120)
        self.lidar_max_range = 6.0
        self.lidar_ray_step = 0
        self._lidar_geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self._lidar_geomid_out = np.zeros(120, dtype=np.int32)
        self._lidar_dist_out = np.zeros(120, dtype=np.float64)
        
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        self._lidar_local_vecs = np.stack([np.cos(angles), -np.sin(angles), np.zeros(120)], axis=1)

        self.kinematic_mode = False
        self._was_running_before_kinematic = False
        self._slider_update_pending = False
        self._running_real_t = 0.0
        
        self.pip_mode = "RGB"
        self.imu_noise = 0.0
        self.lidar_noise = 0.0

        self.timeline_hist = deque(maxlen=300)
        self.is_scrubbing = False
        
        self.spawn_idx = 0
        self.max_spawns = 3
        
        self.cmd_queue = queue.Queue()
        self.state_queue = queue.Queue()
        self.api_server = None

        self.waypoints = []
        self.path_pts = []
        self.path_idx = 0
        
        self.is_grabbing = False
        self.grabbed_bid = -1
        self.grab_offset_pos = np.zeros(3)
        self.grab_offset_quat = np.zeros(4)
        
        self.cop_xy = np.zeros(2)
        self.cop_valid = False
        self.support_poly_pts = []
        self.zmp_inside = True

        # Undo/Redo
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        
        # Collisions
        self.selected_coll_geoms = []

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._populate_actuators()
        self._populate_tree()
        self._restore_settings()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

        self._clock = QElapsedTimer(); self._clock.start()
        self._last_real_t = 0.0; self._fps_count = 0; self._fps_acc = 0.0; self._fps = 0.0
        self._coll_update_timer = 0

        self.log(f"[INFO] Environment loaded: {self.model.modelname}")
        self.log("[INFO] Pose Studio initialized. UI Ready.")

    def _rebuild_caches(self):
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.vel_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_vel")
        self.gyro_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.height_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "height_sensor")
        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        self.gripper_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.finger_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "finger_l")
        self.finger_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "finger_r")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    def _build_ui(self):
        self.viewport = MuJoCoViewport(self.model, self.data, self)
        self.setCentralWidget(self.viewport)

        self.dock_left = QDockWidget("Scene Explorer", self)
        w_left = QWidget(); lay_left = QVBoxLayout(w_left)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search bodies/joints...")
        self.search_box.textChanged.connect(self._filter_tree)
        lay_left.addWidget(self.search_box)

        splitter_left = QSplitter(Qt.Vertical)

        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        splitter_left.addWidget(self.tree)

        g_tel = QGroupBox("Robot Base State")
        l_tel = QFormLayout(g_tel)
        self.lbl_pos = QLabel("—"); self.lbl_vel = QLabel("—")
        self.lbl_battery = QLabel("100%")
        for w_ in (self.lbl_pos, self.lbl_vel, self.lbl_battery):
            w_.setStyleSheet("font-family: Consolas;")
        l_tel.addRow("Pos (X,Y,Z):", self.lbl_pos)
        l_tel.addRow("Speed (m/s):", self.lbl_vel)
        l_tel.addRow("Battery:", self.lbl_battery)
        splitter_left.addWidget(g_tel)

        self.g_props = QGroupBox("Selection Properties (Live Edit)")
        self.l_props = QFormLayout(self.g_props)
        self.lbl_sel_name = QLabel("None")
        self.lbl_sel_pos = QLabel("—"); self.lbl_sel_vel = QLabel("—")
        self.lbl_sel_mass = QLabel("—"); self.lbl_sel_quat = QLabel("—")
        
        self.spn_mass = QDoubleSpinBox(); self.spn_mass.setRange(0.01, 100.0)
        self.spn_mass.valueChanged.connect(self._edit_selected_mass)
        self.spn_fric = QDoubleSpinBox(); self.spn_fric.setRange(0.0, 5.0); self.spn_fric.setSingleStep(0.1)
        self.spn_fric.valueChanged.connect(self._edit_selected_friction)
        
        for w_ in (self.lbl_sel_pos, self.lbl_sel_vel, self.lbl_sel_mass, self.lbl_sel_quat):
            w_.setStyleSheet("font-family: Consolas;")
        self.l_props.addRow("Name:", self.lbl_sel_name)
        self.l_props.addRow("Pos:", self.lbl_sel_pos); self.l_props.addRow("Quat:", self.lbl_sel_quat)
        self.l_props.addRow("Lin Vel:", self.lbl_sel_vel)
        self.l_props.addRow("Edit Mass (kg):", self.spn_mass)
        self.l_props.addRow("Edit Friction:", self.spn_fric)
        splitter_left.addWidget(self.g_props)

        g_imu = QGroupBox("IMU Data (3-axis)")
        l_imu = QVBoxLayout(g_imu); self.imu_graph = IMUGraph(); l_imu.addWidget(self.imu_graph)
        splitter_left.addWidget(g_imu)

        g_com = QGroupBox("CoM Tracker")
        l_com = QVBoxLayout(g_com); self.com_plot = CoMPlot(); l_com.addWidget(self.com_plot)
        splitter_left.addWidget(g_com)

        splitter_left.setSizes([200, 100, 150, 100, 100])
        lay_left.addWidget(splitter_left)
        self.dock_left.setWidget(w_left)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_left)

        self.dock_right = QDockWidget("Control Panel", self)
        self.tabs_right = QTabWidget()

        tab1 = QWidget(); tl1 = QVBoxLayout(tab1)
        g_loc = QGroupBox("Locomotion & Teleop")
        gl_loc = QVBoxLayout(g_loc)
        self.chk_walk = QCheckBox("Enable Gait Controller"); self.chk_walk.setChecked(True)
        self.chk_walk.toggled.connect(lambda b: setattr(self, 'gait_enabled', b))
        gl_loc.addWidget(self.chk_walk)

        h_gait = QHBoxLayout(); h_gait.addWidget(QLabel("Gait:"))
        self.cmb_gait = QComboBox()
        self.cmb_gait.addItems(["Trot", "Walk", "Bound", "Pronk"])
        self.cmb_gait.currentTextChanged.connect(lambda t: setattr(self, 'current_gait', t))
        h_gait.addWidget(self.cmb_gait)
        gl_loc.addLayout(h_gait)

        h_freq = QHBoxLayout(); h_freq.addWidget(QLabel("Freq:"))
        s_freq = QSlider(Qt.Horizontal); s_freq.setRange(5, 40); s_freq.setValue(15)
        self.lbl_freq = QLabel("1.5 Hz")
        s_freq.valueChanged.connect(self._on_freq_change)
        h_freq.addWidget(s_freq); h_freq.addWidget(self.lbl_freq)
        gl_loc.addLayout(h_freq)

        g_pd = QGroupBox("Live PD Tuning")
        l_pd = QFormLayout(g_pd)
        self.lbl_kp = QLabel("Kp: 150")
        self.s_kp = QSlider(Qt.Horizontal); self.s_kp.setRange(10, 300); self.s_kp.setValue(150)
        self.s_kp.valueChanged.connect(self._update_pd)
        l_pd.addRow(self.lbl_kp, self.s_kp)
        self.lbl_kv = QLabel("Kv: 5")
        self.s_kv = QSlider(Qt.Horizontal); self.s_kv.setRange(0, 20); self.s_kv.setValue(5)
        self.s_kv.valueChanged.connect(self._update_pd)
        l_pd.addRow(self.lbl_kv, self.s_kv)
        gl_loc.addWidget(g_pd)

        g_push = QGroupBox("Push Recovery Testing")
        l_push = QVBoxLayout(g_push)
        h_push1 = QHBoxLayout()
        b_push_f = QPushButton("Push Fwd"); b_push_f.clicked.connect(lambda: self._apply_push(0, 3.0))
        b_push_b = QPushButton("Push Back"); b_push_b.clicked.connect(lambda: self._apply_push(0, -3.0))
        h_push1.addWidget(b_push_f); h_push1.addWidget(b_push_b)
        l_push.addLayout(h_push1)
        h_push2 = QHBoxLayout()
        b_push_l = QPushButton("Push Left"); b_push_l.clicked.connect(lambda: self._apply_push(1, 3.0))
        b_push_r = QPushButton("Push Right"); b_push_r.clicked.connect(lambda: self._apply_push(1, -3.0))
        h_push2.addWidget(b_push_l); h_push2.addWidget(b_push_r)
        l_push.addLayout(h_push2)
        gl_loc.addWidget(g_push)

        teleop_help = QLabel("W/S: Fwd/Back\nA/D: Turn L/R\nQ/E: Strafe L/R\nShift+LMB: Waypoints\nG: Toggle Grab\nCtrl+Z: Undo")
        teleop_help.setStyleSheet("color: #aaa; font-family: Consolas;")
        gl_loc.addWidget(teleop_help)

        h_nav_btns = QHBoxLayout()
        b_set_nav = QPushButton("🎯 Set Nav Goal"); b_set_nav.clicked.connect(self.viewport._enter_nav_mode)
        b_cancel_nav = QPushButton("Cancel Nav"); b_cancel_nav.clicked.connect(self.cancel_nav)
        b_clear_trail = QPushButton("Clear Trail"); b_clear_trail.clicked.connect(self.clear_trail)
        b_clear_wp = QPushButton("Clear Path"); b_clear_wp.clicked.connect(self.clear_waypoints)
        h_nav_btns.addWidget(b_set_nav); h_nav_btns.addWidget(b_cancel_nav); h_nav_btns.addWidget(b_clear_trail)
        gl_loc.addLayout(h_nav_btns)
        gl_loc.addWidget(b_clear_wp)
        tl1.addWidget(g_loc)

        g_wind = QGroupBox("Wind Disturbance (Gusty, N)")
        l_wind = QFormLayout(g_wind)
        self.lbl_wx = QLabel("0.0 N"); self.lbl_wy = QLabel("0.0 N")
        s_wx = QSlider(Qt.Horizontal); s_wx.setRange(-20, 20); s_wx.setValue(0)
        s_wx.valueChanged.connect(self._on_wind_x_change)
        l_wind.addRow("Wind X:", s_wx); l_wind.addRow("  Value:", self.lbl_wx)
        s_wy = QSlider(Qt.Horizontal); s_wy.setRange(-20, 20); s_wy.setValue(0)
        s_wy.valueChanged.connect(self._on_wind_y_change)
        l_wind.addRow("Wind Y:", s_wy); l_wind.addRow("  Value:", self.lbl_wy)
        tl1.addWidget(g_wind)
        tl1.addStretch()
        self.tabs_right.addTab(tab1, "Locomotion")

        self.tab_joints = QWidget(); self.tab_joints_lay = QVBoxLayout(self.tab_joints)
        self.j_scroll = QScrollArea(); self.j_scroll.setWidgetResizable(True)
        self.j_scroll_content = QWidget(); self.j_scroll_lay = QVBoxLayout(self.j_scroll_content)
        self.j_scroll.setWidget(self.j_scroll_content); self.tab_joints_lay.addWidget(self.j_scroll)
        self.tabs_right.addTab(self.tab_joints, "Actuators")
        
        tab_manip = QWidget(); tl_manip = QVBoxLayout(tab_manip)
        g_grip = QGroupBox("Gripper Control")
        l_grip = QVBoxLayout(g_grip)
        self.btn_grab = QPushButton("🤚 Grab / Release (G)")
        self.btn_grab.setCheckable(True)
        self.btn_grab.toggled.connect(self._toggle_grab)
        l_grip.addWidget(self.btn_grab)
        tl_manip.addWidget(g_grip)
        tl_manip.addStretch()
        self.tabs_right.addTab(tab_manip, "Manipulation")

        tab3 = QWidget(); tl3 = QVBoxLayout(tab3)
        g_poses = QGroupBox("Persistent Pose Library")
        l_poses = QVBoxLayout(g_poses)
        h_default_poses = QHBoxLayout()
        b_home = QPushButton("Home (Stand)"); b_home.clicked.connect(lambda: self.apply_default_pose("home"))
        b_squat = QPushButton("Squat"); b_squat.clicked.connect(lambda: self.apply_default_pose("squat"))
        h_default_poses.addWidget(b_home); h_default_poses.addWidget(b_squat)
        l_poses.addLayout(h_default_poses)

        self.pose_list = QListWidget(); l_poses.addWidget(self.pose_list)
        h_pose_btns = QHBoxLayout()
        b_save_pose = QPushButton("Save Current"); b_save_pose.clicked.connect(self._save_pose)
        b_load_pose = QPushButton("Load Selected"); b_load_pose.clicked.connect(self._load_pose)
        b_del_pose = QPushButton("Delete"); b_del_pose.clicked.connect(self._del_pose)
        h_pose_btns.addWidget(b_save_pose); h_pose_btns.addWidget(b_load_pose); h_pose_btns.addWidget(b_del_pose)
        l_poses.addLayout(h_pose_btns)

        h_json_btns = QHBoxLayout()
        b_export = QPushButton("Export JSON"); b_export.clicked.connect(self._export_poses)
        b_import = QPushButton("Import JSON"); b_import.clicked.connect(self._import_poses)
        h_json_btns.addWidget(b_export); h_json_btns.addWidget(b_import)
        l_poses.addLayout(h_json_btns)
        tl3.addWidget(g_poses)
        self.tabs_right.addTab(tab3, "Poses")

        tab4 = QWidget(); tl4 = QVBoxLayout(tab4)
        g_sens = QGroupBox("Sensor Noise Simulation")
        l_sens = QFormLayout(g_sens)
        self.lbl_imu_n = QLabel("0.00 rad/s")
        s_imu_n = QSlider(Qt.Horizontal); s_imu_n.setRange(0, 100); s_imu_n.setValue(0)
        s_imu_n.valueChanged.connect(self._on_imu_noise)
        l_sens.addRow("IMU Noise:", s_imu_n); l_sens.addRow("  Value:", self.lbl_imu_n)
        
        self.lbl_lid_n = QLabel("0.00 m")
        s_lid_n = QSlider(Qt.Horizontal); s_lid_n.setRange(0, 100); s_lid_n.setValue(0)
        s_lid_n.valueChanged.connect(self._on_lidar_noise)
        l_sens.addRow("LiDAR Noise:", s_lid_n); l_sens.addRow("  Value:", self.lbl_lid_n)
        tl4.addWidget(g_sens)

        g_motor = QGroupBox("Motor Wear & Friction (FL Leg)")
        l_motor = QFormLayout(g_motor)
        self.lbl_mot = QLabel("0.00 Nm")
        s_mot = QSlider(Qt.Horizontal); s_mot.setRange(0, 100); s_mot.setValue(0)
        s_mot.valueChanged.connect(self._on_motor_wear)
        l_motor.addRow("Friction Loss:", s_mot); l_motor.addRow("  Value:", self.lbl_mot)
        tl4.addWidget(g_motor)
        tl4.addStretch()
        self.tabs_right.addTab(tab4, "Sensors & Health")

        self.dock_right.setWidget(self.tabs_right)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_right)

        self.dock_bot = QDockWidget("Tools & Diagnostics", self)
        self.tabs_bot = QTabWidget()
        self.console = QTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet("background: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        self.tabs_bot.addTab(self.console, "Console")

        # Collision Inspector Tab
        self.tree_collisions = QTreeWidget(); self.tree_collisions.setHeaderLabels(["Body 1", "Body 2", "Pen (mm)", "Normal (N)", "Friction (N)"])
        self.tree_collisions.itemSelectionChanged.connect(self._on_collision_select)
        self.tabs_bot.addTab(self.tree_collisions, "Collisions")

        plot_tab = QWidget(); tl_plot = QVBoxLayout(plot_tab)
        self.torque_plot = TorquePlot(self.model.nu); tl_plot.addWidget(self.torque_plot)
        self.tabs_bot.addTab(plot_tab, "Live Plots")

        lidar_tab = QWidget(); tl_lidar = QVBoxLayout(lidar_tab)
        self.lidar_plot = LidarPlot(); tl_lidar.addWidget(self.lidar_plot)
        self.tabs_bot.addTab(lidar_tab, "Lidar Scan")

        self.dock_bot.setWidget(self.tabs_bot)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_bot)
        self.dock_bot.setMaximumHeight(300)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.sliderPressed.connect(self._on_timeline_press)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_release)
        self.timeline_slider.valueChanged.connect(self._on_timeline_change)
        self.dock_bot.setTitleBarWidget(QWidget())
        tb_layout = QVBoxLayout()
        tb_layout.addWidget(self.timeline_slider)
        w_bot = self.dock_bot.widget()
        w_bot.layout().addLayout(tb_layout)

    def _on_collision_select(self):
        items = self.tree_collisions.selectedItems()
        if items:
            self.selected_coll_geoms = [items[0].data(0, Qt.UserRole), items[0].data(1, Qt.UserRole)]

    def _update_collision_tree(self):
        if self.tabs_bot.currentIndex() != 1: return
        self.tree_collisions.clear()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1; g2 = c.geom2
            b1_id = self.model.geom_bodyid[g1]
            b2_id = self.model.geom_bodyid[g2]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1_id) or f"body_{b1_id}"
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2_id) or f"body_{b2_id}"
            
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            fn = f[2]
            ff = math.hypot(f[0], f[1])
            
            item = QTreeWidgetItem([b1, b2, f"{-c.dist*1000:.2f}", f"{fn:.2f}", f"{ff:.2f}"])
            item.setData(0, Qt.UserRole, g1)
            item.setData(1, Qt.UserRole, g2)
            self.tree_collisions.addTopLevelItem(item)

    def push_undo(self, qpos):
        self.redo_stack.clear()
        self.undo_stack.append(qpos.copy())

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.data.qpos.copy())
            self.data.qpos[:] = self.undo_stack.pop()
            mujoco.mj_forward(self.model, self.data)
            self.update_sliders_from_qpos()
            self.log("[INFO] Undo")

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.data.qpos.copy())
            self.data.qpos[:] = self.redo_stack.pop()
            mujoco.mj_forward(self.model, self.data)
            self.update_sliders_from_qpos()
            self.log("[INFO] Redo")

    def _toggle_grab(self, c):
        self.is_grabbing = c
        if c:
            self.log("[INFO] Attempting to grab...")
        else:
            self.log("[INFO] Released grab.")
            self.grabbed_bid = -1

    def _on_imu_noise(self, v):
        self.imu_noise = v / 100.0
        self.lbl_imu_n.setText(f"{self.imu_noise:.2f} rad/s")

    def _on_lidar_noise(self, v):
        self.lidar_noise = v / 100.0
        self.lbl_lid_n.setText(f"{self.lidar_noise:.2f} m")

    def _on_motor_wear(self, v):
        wear = v / 10.0
        self.lbl_mot.setText(f"{wear:.2f} Nm")
        for j_name in ['fl_haa', 'fl_hfe', 'fl_kfe']:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if jid >= 0:
                self.model.dof_frictionloss[self.model.jnt_dofadr[jid]] = wear

    def _edit_selected_mass(self, v):
        items = self.tree.selectedItems()
        if items and items[0].text(1) == "body":
            bid = items[0].data(0, Qt.UserRole)
            if bid is not None and bid > 0:
                self.model.body_mass[bid] = float(v)
                mujoco.mj_setConst(self.model, self.data)

    def _edit_selected_friction(self, v):
        items = self.tree.selectedItems()
        if items and items[0].text(1) == "body":
            bid = items[0].data(0, Qt.UserRole)
            if bid is not None and bid > 0:
                g_adr = self.model.body_geomadr[bid]
                g_num = self.model.body_geomnum[bid]
                for g in range(g_num):
                    self.model.geom_friction[g_adr + g][0] = float(v)

    def _on_timeline_press(self):
        self.is_scrubbing = True
        if self.sim_running: self._toggle_play()

    def _on_timeline_release(self):
        self.is_scrubbing = False

    def _on_timeline_change(self, val):
        if self.is_scrubbing and self.timeline_hist:
            idx = min(val, len(self.timeline_hist)-1)
            qpos, qvel = self.timeline_hist[idx]
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            mujoco.mj_forward(self.model, self.data)

    def _on_freq_change(self, v):
        self.gait_freq = v / 10.0
        self.lbl_freq.setText(f"{self.gait_freq:.1f} Hz")

    def _on_wind_x_change(self, v):
        self.wind_x = float(v)
        self.lbl_wx.setText(f"{self.wind_x:.1f} N")

    def _on_wind_y_change(self, v):
        self.wind_y = float(v)
        self.lbl_wy.setText(f"{self.wind_y:.1f} N")

    def _update_pd(self, _):
        kp = self.s_kp.value()
        kv = self.s_kv.value()
        self.model.actuator_gainprm[:, 0] = float(kp)
        self.model.actuator_biasprm[:, 1] = -float(kp)
        self.model.actuator_biasprm[:, 2] = -float(kv)
        self.lbl_kp.setText(f"Kp: {kp}")
        self.lbl_kv.setText(f"Kv: {kv}")

    def _apply_push(self, axis, vel):
        if self.trunk_id >= 0:
            qvel_adr = self.model.jnt_dofadr[self.model.body_jntadr[self.trunk_id]]
            self.data.qvel[qvel_adr + axis] += vel
            self.log(f"[IMPULSE] Applied {vel} m/s to axis {axis}")

    def _filter_tree(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            self._filter_item(item, text)

    def _filter_item(self, item, text, parent_visible=True):
        self_match = text in item.text(0).lower()
        any_child_match = False
        for i in range(item.childCount()):
            if self._filter_item(item.child(i), text, parent_visible and self_match):
                any_child_match = True
        visible = parent_visible and (self_match or any_child_match)
        item.setHidden(not visible)
        return self_match or any_child_match

    def _populate_tree(self):
        self.tree.clear()
        self._body_items = {}  
        root = QTreeWidgetItem(["World", "world"])
        self.tree.addTopLevelItem(root)
        self._body_items[0] = root
        items = {0: root}
        for i in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            pid = int(self.model.body_parentid[i])
            parent = items.get(pid, root)
            item = QTreeWidgetItem(parent, [name, "body"])
            item.setData(0, Qt.UserRole, i)
            item.setForeground(0, QColor(100, 180, 255))
            items[i] = item
            self._body_items[i] = item
            jnt_adr = self.model.body_jntadr[i]; jnt_num = self.model.body_jntnum[i]
            for j in range(jnt_num):
                jid = jnt_adr + j
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
                j_item = QTreeWidgetItem(item, [jname, "joint"])
                j_item.setForeground(0, QColor(255, 200, 100))
            geom_adr = self.model.body_geomadr[i]; geom_num = self.model.body_geomnum[i]
            for g in range(geom_num):
                gid = geom_adr + g
                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                g_item = QTreeWidgetItem(item, [gname, "geom"])
                g_item.setForeground(0, QColor(180, 180, 180))
        self.tree.expandAll()

    def select_body_in_tree(self, bid):
        item = self._body_items.get(bid)
        if item is not None:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

    def _on_tree_select(self):
        items = self.tree.selectedItems()
        if not items: return
        item = items[0]
        self.lbl_sel_name.setText(item.text(0))
        if item.text(1) == "body":
            bid = item.data(0, Qt.UserRole)
            if bid is not None and bid > 0:
                p = self.data.xpos[bid]
                q = self.data.xquat[bid]
                self.lbl_sel_pos.setText(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
                self.lbl_sel_quat.setText(f"{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}")
                self.lbl_sel_mass.setText(f"{self.model.body_mass[bid]:.3f} kg")
                self.g_props.setTitle(f"Properties: {item.text(0)}")
                self.spn_mass.blockSignals(True)
                self.spn_mass.setValue(self.model.body_mass[bid])
                self.spn_mass.blockSignals(False)

    def _populate_actuators(self):
        for i in reversed(range(self.j_scroll_lay.count())):
            w_ = self.j_scroll_lay.itemAt(i).widget()
            if w_ is not None: w_.setParent(None)
        self.j_sliders = []; self.j_act_map = {}; self.j_lbls = {}
        groups = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            prefix = name.split('_')[0]
            if prefix not in groups: groups[prefix] = []
            groups[prefix].append((i, name))
        for prefix, acts in groups.items():
            g_box = QGroupBox(f"{prefix.upper()} Joints")
            g_lay = QVBoxLayout(g_box)
            for idx, name in acts:
                row = QHBoxLayout()
                lbl_name = QLabel(name); lbl_name.setMinimumWidth(80)
                jid = self.model.actuator_trnid[idx, 0]
                rng = self.model.jnt_range[jid]
                min_deg = int(math.degrees(rng[0])); max_deg = int(math.degrees(rng[1]))
                s = QSlider(Qt.Horizontal); s.setRange(min_deg, max_deg); s.setValue(0)
                s.setProperty("act_idx", idx)
                v_lbl = QLabel("T:0° A:0°  F:0.0N"); v_lbl.setMinimumWidth(130)
                s.valueChanged.connect(self._on_slider_changed)
                row.addWidget(lbl_name); row.addWidget(s); row.addWidget(v_lbl)
                g_lay.addLayout(row)
                self.j_sliders.append(s); self.j_act_map[s] = idx; self.j_lbls[s] = v_lbl
            self.j_scroll_lay.addWidget(g_box)
        self.j_scroll_lay.addStretch()

    def _on_slider_changed(self, val):
        idx = self.sender().property("act_idx")
        if self.kinematic_mode:
            self.data.qpos[self.act_qpos[idx]] = math.radians(val)
            mujoco.mj_forward(self.model, self.data)
        else:
            self.manual_targets[idx] = math.radians(val)
        lbl = self.j_lbls[self.sender()]
        lbl.setText(lbl.text().split('A')[0] + f"A:{val}°  F:0.0N")

    def update_sliders_from_qpos(self):
        if self._slider_update_pending:
            return
        self._slider_update_pending = True
        QTimer.singleShot(0, self._do_slider_update)

    def _do_slider_update(self):
        self._slider_update_pending = False
        for s, idx in self.j_act_map.items():
            jid = self.model.actuator_trnid[idx, 0]
            rng = self.model.jnt_range[jid]
            min_deg = int(math.degrees(rng[0])); max_deg = int(math.degrees(rng[1]))
            target_val = int(math.degrees(self.manual_targets[idx]))
            actual_val = int(math.degrees(self.data.qpos[self.act_qpos[idx]]))
            
            if self.kinematic_mode:
                target_val = actual_val
                
            force_val = float(self.data.actuator_force[idx]) if self.model.nu > 0 else 0.0
            target_val = max(min_deg, min(max_deg, target_val))
            actual_val = max(min_deg, min(max_deg, actual_val))
            s.blockSignals(True)
            if s.value() != target_val:
                s.setValue(target_val)
            self.j_lbls[s].setText(f"T:{target_val:3d}° A:{actual_val:3d}°  F:{force_val:5.1f}N")
            s.blockSignals(False)

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a_snap = QAction("📷 Save Snapshot", self); a_snap.triggered.connect(self._save_snapshot)
        m_file.addAction(a_snap); m_file.addSeparator()
        a_exit = QAction("Exit", self); a_exit.triggered.connect(self.close); m_file.addAction(a_exit)

        m_edit = mb.addMenu("&Edit")
        a_undo = QAction("Undo (Ctrl+Z)", self); a_undo.triggered.connect(self.undo)
        m_edit.addAction(a_undo)
        a_redo = QAction("Redo (Ctrl+Y)", self); a_redo.triggered.connect(self.redo)
        m_edit.addAction(a_redo)

        m_sim = mb.addMenu("&Simulation")
        self.a_play = QAction("&Play", self)
        self.a_play.setShortcut("Space")
        self.a_play.triggered.connect(self._toggle_play); m_sim.addAction(self.a_play)
        a_reset = QAction("&Reset All", self); a_reset.setShortcut("Ctrl+R")
        a_reset.triggered.connect(self._reset_sim); m_sim.addAction(a_reset)
        a_reset_robot = QAction("Reset Robot &Only", self); a_reset_robot.setShortcut("Ctrl+Shift+R")
        a_reset_robot.triggered.connect(self.reset_robot_only); m_sim.addAction(a_reset_robot)
        m_sim.addSeparator()
        self.a_kinematic = QAction("Skeleton Mode (Kinematic)", self, checkable=True)
        self.a_kinematic.setShortcut("K")
        self.a_kinematic.triggered.connect(self._toggle_kinematic); m_sim.addAction(self.a_kinematic)

        m_view = mb.addMenu("&View")
        
        m_layout = m_view.addMenu("Viewport Layout")
        self.g_layout = QActionGroup(self)
        for mode, val in [("1 Viewport", 1), ("2 Viewports (Split)", 2), ("4 Viewports (Quad)", 4)]:
            a = QAction(mode, self, checkable=True)
            if val == 1: a.setChecked(True)
            a.triggered.connect(lambda c, v=val: self._set_view_layout(v))
            self.g_layout.addAction(a)
            m_layout.addAction(a)

        self.a_track = QAction("Track Robot", self, checkable=True)
        self.a_track.triggered.connect(self._toggle_track); m_view.addAction(self.a_track)
        self.a_chase_cam = QAction("Chase Camera (Yaw Follow)", self, checkable=True)
        self.a_chase_cam.triggered.connect(self._toggle_chase_cam); m_view.addAction(self.a_chase_cam)
        self.a_skeleton_view = QAction("Show Skeleton (Transparent)", self, checkable=True)
        self.a_skeleton_view.triggered.connect(self._toggle_skeleton_view); m_view.addAction(self.a_skeleton_view)
        
        m_pip = m_view.addMenu("PiP Camera Mode")
        self.g_pip = QActionGroup(self)
        for mode in ["RGB", "Segmentation", "Depth"]:
            a = QAction(mode, self, checkable=True)
            if mode == "RGB": a.setChecked(True)
            a.triggered.connect(lambda c, m=mode: self._set_pip_mode(m))
            self.g_pip.addAction(a)
            m_pip.addAction(a)

        self.a_show_frames = QAction("Show Coord Frames", self, checkable=True)
        self.a_show_frames.triggered.connect(self._toggle_frames); m_view.addAction(self.a_show_frames)
        self.a_show_com = QAction("Show Center of Mass", self, checkable=True)
        self.a_show_com.triggered.connect(self._toggle_com); m_view.addAction(self.a_show_com)
        self.a_show_inertia = QAction("Show Inertia", self, checkable=True)
        self.a_show_inertia.triggered.connect(self._toggle_inertia); m_view.addAction(self.a_show_inertia)
        m_view.addSeparator()
        self.a_toggle_left = QAction("Toggle Left Dock", self); self.a_toggle_left.setShortcut("Alt+1")
        self.a_toggle_left.triggered.connect(lambda: self.dock_left.setVisible(not self.dock_left.isVisible()))
        m_view.addAction(self.a_toggle_left)
        self.a_toggle_right = QAction("Toggle Right Dock", self); self.a_toggle_right.setShortcut("Alt+2")
        self.a_toggle_right.triggered.connect(lambda: self.dock_right.setVisible(not self.dock_right.isVisible()))
        m_view.addAction(self.a_toggle_right)
        self.a_toggle_bot = QAction("Toggle Bottom Dock", self); self.a_toggle_bot.setShortcut("Alt+3")
        self.a_toggle_bot.triggered.connect(lambda: self.dock_bot.setVisible(not self.dock_bot.isVisible()))
        m_view.addAction(self.a_toggle_bot)
        
        m_terrain = mb.addMenu("&Terrain")
        a_flat = QAction("Flat Ground", self); a_flat.triggered.connect(lambda: self._generate_terrain("flat"))
        m_terrain.addAction(a_flat)
        a_rough = QAction("Rough Terrain (Procedural)", self); a_rough.triggered.connect(lambda: self._generate_terrain("rough"))
        m_terrain.addAction(a_rough)
        a_stairs = QAction("Stairs", self); a_stairs.triggered.connect(lambda: self._generate_terrain("stairs"))
        m_terrain.addAction(a_stairs)
        
        m_spawn = mb.addMenu("&Spawning")
        a_box = QAction("Spawn Box", self); a_box.triggered.connect(lambda: self._spawn_obj("box"))
        m_spawn.addAction(a_box)
        a_sph = QAction("Spawn Sphere", self); a_sph.triggered.connect(lambda: self._spawn_obj("sphere"))
        m_spawn.addAction(a_sph)
        
        m_api = mb.addMenu("&Network API")
        self.a_api = QAction("Start TCP API Server (Port 5555)", self, checkable=True)
        self.a_api.triggered.connect(self._toggle_api_server)
        m_api.addAction(self.a_api)
        
        m_help = mb.addMenu("&Help")
        a_help = QAction("Controls &Shortcuts", self)
        a_help.triggered.connect(self._show_help)
        m_help.addAction(a_help)

    def _set_view_layout(self, layout):
        self.viewport.view_layout = layout
        self.viewport.update()

    def _set_pip_mode(self, mode):
        self.pip_mode = mode
        if mode == "RGB":
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_RGBA] = False
        elif mode == "Segmentation":
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_RGBA] = True
        elif mode == "Depth":
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
            self.viewport.pip_opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False

    def _generate_terrain(self, terrain_type):
        self.log(f"[INFO] Generating terrain: {terrain_type}")
        if terrain_type == "flat":
            xml = ENV_FLAT
        elif terrain_type == "rough":
            res = 12
            size = 5.0
            verts = []
            faces = []
            for i in range(res):
                for j in range(res):
                    x = (i / (res-1) - 0.5) * size
                    y = (j / (res-1) - 0.5) * size
                    z = random.uniform(0, 0.15) if (i > 0 and j > 0 and i < res-1 and j < res-1) else 0
                    verts.append(f"{x} {y} {z}")
            for i in range(res-1):
                for j in range(res-1):
                    v0 = i * res + j + 1
                    v1 = i * res + j + 2
                    v2 = (i + 1) * res + j + 1
                    v3 = (i + 1) * res + j + 2
                    faces.append(f"{v0} {v2} {v1}")
                    faces.append(f"{v1} {v2} {v3}")
            xml = ENV_FLAT.replace(
                '<geom name="ground" type="plane" size="20 20 0.1" material="grid_mat" group="1"/>',
                f'<mesh name="rough_mesh" vertex="{chr(10).join([f"v {v}" for v in verts])}" face="{chr(10).join([f"f {f}" for f in faces])}"/>\n'
                f'<geom name="ground" type="mesh" mesh="rough_mesh" material="grid_mat" group="1"/>'
            )
        elif terrain_type == "stairs":
            xml = ENV_FLAT
            
        self.current_env_xml = xml
        self._reload_model()

    def _spawn_obj(self, obj_type):
        if self.spawn_idx >= self.max_spawns:
            self.log("[WARN] Max spawn objects reached.")
            return
            
        name = f"spawn_{obj_type}_{self.spawn_idx}"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            mid = self.model.body_mocapid[bid]
            trunk_pos = self.data.xpos[self.trunk_id]
            self.data.mocap_pos[mid] = [trunk_pos[0] + random.uniform(-0.5, 0.5), trunk_pos[1] + random.uniform(-0.5, 0.5), 1.5]
            self.log(f"[INFO] Spawned {name}")
            self.spawn_idx += 1

    def _toggle_api_server(self, checked):
        if checked:
            self.api_server = APIServerThread(5555, self.cmd_queue, self.state_queue)
            self.api_server.start()
            self.log("[INFO] TCP API Server started on port 5555.")
        else:
            if self.api_server:
                self.api_server.stop()
                self.api_server = None
            self.log("[INFO] TCP API Server stopped.")

    def _reload_model(self):
        self.model = mujoco.MjModel.from_xml_string(self.current_env_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.act_qpos = np.zeros(self.model.nu, dtype=int)
        self.act_qvel = np.zeros(self.model.nu, dtype=int)
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            self.act_qpos[i] = self.model.jnt_qposadr[jid]
            self.act_qvel[i] = self.model.jnt_dofadr[jid]
        self._rebuild_caches()
        self.viewport.update_model(self.model, self.data)
        self._populate_tree()
        self._populate_actuators()
        self.spawn_idx = 0
        self.manual_targets[:] = 0.0
        self.timeline_hist.clear()
        self.timeline_slider.setMaximum(0)
        self.clear_waypoints()

    def clear_waypoints(self):
        self.waypoints.clear()
        self.path_pts.clear()
        self.path_idx = 0
        self.log("[INFO] Waypoints cleared.")

    def add_waypoint(self, xy):
        self.waypoints.append(xy)
        self.nav_goal = None 
        if len(self.waypoints) >= 2:
            self._generate_spline()
            self.log(f"[NAV] Waypoint added. Total: {len(self.waypoints)}. Path generated.")
        else:
            self.log(f"[NAV] Waypoint added. Need at least 2 to generate path.")

    def _generate_spline(self):
        pts = self.waypoints
        if len(pts) < 2: return
        self.path_pts = []
        p_dummy = [2 * np.array(pts[0]) - np.array(pts[1])]
        p_dummy.extend([np.array(p) for p in pts])
        p_dummy.append(2 * np.array(pts[-1]) - np.array(pts[-2]))
        
        num_segments = 20
        for i in range(len(p_dummy) - 3):
            p0, p1, p2, p3 = p_dummy[i], p_dummy[i+1], p_dummy[i+2], p_dummy[i+3]
            for t in np.linspace(0, 1, num_segments):
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t2 + (-p0 + 3*p1 - 3*p2 + p3) * t3)
                self.path_pts.append(tuple(pt))
        self.path_idx = 0

    def _show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("POSE STUDIO Controls")
        msg.setText(
            "=== Mouse Controls ===\n"
            "LMB Drag: Orbit Camera\n"
            "RMB Drag: Zoom Camera\n"
            "MMB Drag: Pan Camera\n"
            "Ctrl + LMB: Select Body\n"
            "Shift + LMB: Place Waypoint\n"
            "RMB Click: Context Menu\n\n"
            "=== Keyboard Controls ===\n"
            "W/S: Move Forward/Backward\n"
            "A/D: Turn Left/Right\n"
            "Q/E: Strafe Left/Right\n"
            "G: Toggle Gripper Grab\n"
            "Ctrl+Z: Undo (Skeleton Mode)\n"
            "Ctrl+Y: Redo (Skeleton Mode)\n"
            "Space: Play/Pause Simulation\n"
            "C: Reset Camera\n"
            "Ctrl+R: Reset All\n"
            "K: Toggle Skeleton (Kinematic) Mode\n"
            "Esc: Cancel Nav Goal / Exit\n\n"
            "=== Skeleton Mode ===\n"
            "Drag Trunk: Move Base\n"
            "Alt+Drag Trunk: Rotate Base\n"
            "Drag Foot (Calf): IK Leg Targeting"
        )
        msg.exec()

    def _toggle_chase_cam(self, c):
        self.chase_cam = c
        if c and self.a_track.isChecked():
            self.a_track.setChecked(False)
            self._toggle_track(False)
        self.log(f"[INFO] Chase camera {'enabled' if c else 'disabled'}.")

    def _toggle_kinematic(self, c):
        self.kinematic_mode = c
        if c:
            self._saved_gravity = self.model.opt.gravity.copy()
            self._was_running_before_kinematic = self.sim_running
            self.sim_running = False
            self.btn_play.setText("▶  Play"); self.a_play.setText("&Play")
            self.model.opt.gravity[:] = 0
            self.data.qvel[:] = 0
            self.gait_enabled = False
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.log("[INFO] Skeleton Mode ACTIVE. Sim paused, gravity off.")
        else:
            if hasattr(self, '_saved_gravity'):
                self.model.opt.gravity[:] = self._saved_gravity
            self.manual_targets[:] = self.data.qpos[7:7 + self.model.nu]
            self.gait_enabled = self.chk_walk.isChecked()
            self.log("[INFO] Skeleton Mode DISABLED. Physics restored.")
            if self._was_running_before_kinematic:
                self.sim_running = True
                self.btn_play.setText("⏸  Pause"); self.a_play.setText("&Pause")

    def _toggle_skeleton_view(self, c):
        self.viewport.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = c

    def _toggle_frames(self, c):
        self.viewport.opt.frame = mujoco.mjtFrame.mjFRAME_BODY if c else mujoco.mjtFrame.mjFRAME_NONE

    def _toggle_com(self, c):
        self.viewport.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = c

    def _toggle_inertia(self, c):
        self.viewport.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = c

    def _toggle_track(self, c):
        if c and self.a_chase_cam.isChecked():
            self.a_chase_cam.setChecked(False)
            self._toggle_chase_cam(False)
        if c:
            self.viewport.cam_main.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewport.cam_main.trackbodyid = self.trunk_id; self.viewport.cam_main.distance = 2.0
        else:
            self.viewport.cam_main.type = mujoco.mjtCamera.mjCAMERA_FREE

    def _build_toolbar(self):
        tb = QToolBar("Main"); tb.setMovable(False); tb.setObjectName("MainToolbar")
        self.addToolBar(tb)
        self.btn_play = QPushButton("▶  Play"); self.btn_play.setObjectName("PlayBtn")
        self.btn_play.clicked.connect(self._toggle_play); tb.addWidget(self.btn_play)
        b_reset = QPushButton("⏮  Reset All"); b_reset.clicked.connect(self._reset_sim); tb.addWidget(b_reset)
        b_reset_robot = QPushButton("🤖 Reset Robot"); b_reset_robot.clicked.connect(self.reset_robot_only); tb.addWidget(b_reset_robot)
        self.btn_rec = QPushButton("⏺ Record"); self.btn_rec.setObjectName("RecBtn")
        self.btn_rec.clicked.connect(self._toggle_record); tb.addWidget(self.btn_rec)
        tb.addSeparator(); tb.addWidget(QLabel("  Speed:  "))
        s_speed = QSlider(Qt.Horizontal); s_speed.setRange(1, 100); s_speed.setValue(10); s_speed.setFixedWidth(140)
        s_speed.valueChanged.connect(self._on_speed_change); tb.addWidget(s_speed)
        self.lbl_speed = QLabel("1.0×"); tb.addWidget(self.lbl_speed)

    def _toggle_record(self):
        if not self.recording:
            folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Frames")
            if folder:
                self.rec_folder = folder; self.recording = True; self.rec_frame = 0
                self.btn_rec.setText("⏹ Stop Rec")
                self.btn_rec.setStyleSheet("background: #8b0000; font-weight: bold;")
                self.log(f"[INFO] Recording started. Saving to {self.rec_folder}")
        else:
            self.recording = False; self.btn_rec.setText("⏺ Record"); self.btn_rec.setStyleSheet("")
            self.log(f"[INFO] Recording stopped. Saved {self.rec_frame} frames.")

    def _on_speed_change(self, v):
        val = v / 10.0; self.sim_speed = val; self.lbl_speed.setText(f"{val:.1f}×")
        self.lbl_status_speed.setText(f"Speed: {val:.1f}×")

    def _build_statusbar(self):
        sb = QStatusBar(); self.setStatusBar(sb)
        self.lbl_time = QLabel("t = 0.000 s"); self.lbl_rtf = QLabel("RTF: 0.0x")
        self.lbl_fps = QLabel("FPS: 0"); self.lbl_ncon = QLabel("contacts: 0")
        self.lbl_status_speed = QLabel("Speed: 1.0×")
        for w in (self.lbl_time, self.lbl_rtf, self.lbl_fps, self.lbl_ncon, self.lbl_status_speed):
            sb.addPermanentWidget(w)

    def log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{ts}] {msg}")

    def _toggle_play(self):
        if self.kinematic_mode:
            self.log("[WARN] Cannot play in Skeleton Mode. Disable it first.")
            return
        self.sim_running = not self.sim_running
        if self.sim_running:
            self.btn_play.setText("⏸  Pause"); self.a_play.setText("&Pause"); self.log("[INFO] Simulation running.")
        else:
            self.btn_play.setText("▶  Play"); self.a_play.setText("&Play"); self.log("[INFO] Simulation paused.")

    def clear_trail(self):
        for mid in self.trail_mocap_ids:
            if mid >= 0:
                self.data.mocap_pos[mid] = [10, 10, 10]
        self.trail_idx = 0
        self.log("[INFO] Trail cleared.")

    def _reset_sim(self):
        mujoco.mj_resetData(self.model, self.data); mujoco.mj_forward(self.model, self.data)
        self.sim_time = 0.0; self.manual_targets[:] = 0.0
        self.target_vx = self.target_vy = self.target_wz = 0.0
        self.cmd_vx = self.cmd_vy = self.cmd_wz = 0.0
        self.battery_percent = 100.0
        self.obstacle_blocked = False
        self.recovering = False; self.recover_timer = 0.0
        self._battery_dead_logged = False
        self._running_real_t = 0.0
        self.cancel_nav(); self.clear_trail(); self.clear_waypoints()
        self.timeline_hist.clear()
        self.timeline_slider.setMaximum(0)
        self.spawn_idx = 0
        self.is_grabbing = False
        self.btn_grab.setChecked(False)
        self.update_sliders_from_qpos()
        self.log("[INFO] Simulation reset.")

    def reset_robot_only(self):
        if self.trunk_id >= 0:
            qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[self.trunk_id]]
            qvel_adr = self.model.jnt_dofadr[self.model.body_jntadr[self.trunk_id]]
            self.data.qpos[qpos_adr:qpos_adr+7] = [0, 0, 0.45, 1, 0, 0, 0]
            self.data.qvel[qvel_adr:qvel_adr+6] = 0
            for i in range(self.model.nu):
                jid = self.model.actuator_trnid[i, 0]
                self.data.qpos[self.model.jnt_qposadr[jid]] = 0
                self.data.qvel[self.model.jnt_dofadr[jid]] = 0
            mujoco.mj_forward(self.model, self.data)
            self.manual_targets[:] = 0.0
            self.obstacle_blocked = False
            self.recovering = False
            self.recover_timer = 0.0
            self.target_vx = self.target_vy = self.target_wz = 0.0
            self.cmd_vx = self.cmd_vy = self.cmd_wz = 0.0
            self.cancel_nav(); self.clear_trail()
            self.update_sliders_from_qpos()
            self.log("[INFO] Robot reset to origin. Environment kept intact.")

    def _save_snapshot(self):
        img = self.viewport._image
        if img and not img.isNull():
            path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "robot_snapshot.png", "PNG Files (*.png)")
            if path: img.save(path); self.log(f"[INFO] Snapshot saved to {path}")

    def set_nav_goal(self, xy):
        self.nav_goal = xy
        self.clear_waypoints()
        if self.goal_mocap_id >= 0:
            self.data.mocap_pos[self.goal_mocap_id] = [xy[0], xy[1], 0.1]
        self.log(f"[NAV] Goal set to ({xy[0]:.2f}, {xy[1]:.2f})")

    def cancel_nav(self):
        if self.nav_goal is not None:
            self.nav_goal = None
            if self.goal_mocap_id >= 0:
                self.data.mocap_pos[self.goal_mocap_id] = [10, 10, 10]
            self.log("[NAV] Navigation cancelled.")

    def _save_pose(self):
        name, ok = QInputDialog.getText(self, "Save Pose", "Pose Name:")
        if ok and name:
            self.saved_poses[name] = np.copy(self.data.qpos[7:]).tolist()
            if self.pose_list.findItems(name, Qt.MatchExactly):
                self.log(f"[POSE] Overwrote '{name}'")
            else:
                self.pose_list.addItem(name)
                self.log(f"[POSE] Saved '{name}'")

    def _load_pose(self):
        item = self.pose_list.currentItem()
        if item and item.text() in self.saved_poses:
            self.data.qpos[7:] = self.saved_poses[item.text()]
            self.data.qvel[:] = 0
            self.manual_targets[:] = self.data.qpos[7:7 + self.model.nu]
            mujoco.mj_forward(self.model, self.data)
            self.update_sliders_from_qpos()
            self.log(f"[POSE] Loaded '{item.text()}'")

    def _del_pose(self):
        item = self.pose_list.currentItem()
        if item:
            name = item.text()
            if name in self.saved_poses: del self.saved_poses[name]
            self.pose_list.takeItem(self.pose_list.row(item))
            self.log(f"[POSE] Deleted '{name}'")

    def _export_poses(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Poses", "poses.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(self.saved_poses, f, indent=4)
            self.log(f"[POSE] Exported {len(self.saved_poses)} poses to {path}")

    def _import_poses(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Poses", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    imported = json.load(f)
                for name, pose in imported.items():
                    self.saved_poses[name] = pose
                    if not self.pose_list.findItems(name, Qt.MatchExactly):
                        self.pose_list.addItem(name)
                self.log(f"[POSE] Imported {len(imported)} poses from {path}")
            except Exception as e:
                self.log(f"[ERROR] Failed to import poses: {e}")

    def _leg_actuator_indices(self):
        legs = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            parts = name.split('_')
            if len(parts) == 2:
                leg, joint = parts[0], parts[1]
                legs.setdefault(leg, {})[joint] = i
        return {l: (d['haa'], d['hfe'], d['kfe']) for l, d in legs.items()
                if {'haa','hfe','kfe'} <= d.keys()}

    def apply_default_pose(self, pose_name):
        if pose_name == "home":
            self.data.qpos[0:3] = [0, 0, 0.45]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qpos[7:] = 0.0
        elif pose_name == "squat":
            self.data.qpos[0:3] = [0, 0, 0.25]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qpos[7:] = 0.0
            for haa_i, hfe_i, kfe_i in self._leg_actuator_indices().values():
                self.data.qpos[self.act_qpos[haa_i]] = 0.0
                self.data.qpos[self.act_qpos[hfe_i]] = 1.0
                self.data.qpos[self.act_qpos[kfe_i]] = -2.0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.update_sliders_from_qpos()
        self.log(f"[POSE] Applied default pose: {pose_name}")

    def cast_lidar(self):
        if self.lidar_site_id < 0: return
        pos = self.data.site_xpos[self.lidar_site_id]
        mat = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        global_vecs = (self._lidar_local_vecs @ mat.T).flatten()
        if hasattr(mujoco, 'mj_multiRay'):
            mujoco.mj_multiRay(self.model, self.data, pos, global_vecs, self._lidar_geomgroup,
                               1, -1, self._lidar_geomid_out, self._lidar_dist_out, 120, self.lidar_max_range)
            ranges = np.where((self._lidar_dist_out > 0) & (self._lidar_dist_out < self.lidar_max_range), self._lidar_dist_out, self.lidar_max_range)
        else:
            ranges = np.zeros(120)
            for i in range(120):
                dist = mujoco.mj_ray(self.model, self.data, pos, global_vecs[i*3:i*3+3], self._lidar_geomgroup,
                                     mujoco.mjtObj.mjOBJ_GEOM, -1)
                ranges[i] = dist if 0 < dist < self.lidar_max_range else self.lidar_max_range
        
        if self.lidar_noise > 0:
            ranges += np.random.normal(0, self.lidar_noise, 120)
            ranges = np.clip(ranges, 0, self.lidar_max_range)
            
        self.lidar_ranges = ranges
        self.lidar_plot.update_scan(ranges)

    def _gait_targets(self, t):
        omega = 2 * math.pi * self.gait_freq
        gaits = {
            "Trot": {'fl': 0.0, 'fr': math.pi, 'hl': math.pi, 'hr': 0.0},
            "Walk": {'fl': 0.0, 'fr': 3*math.pi/2, 'hl': math.pi, 'hr': math.pi/2},
            "Bound": {'fl': 0.0, 'fr': 0.0, 'hl': math.pi, 'hr': math.pi},
            "Pronk": {'fl': 0.0, 'fr': 0.0, 'hl': 0.0, 'hr': 0.0}
        }
        phases = gaits[self.current_gait]
        tg = np.zeros(self.model.nu)
        tg[0] = self.cmd_wz * 0.5; tg[1] = self.cmd_vx * 0.2
        i = 2
        for leg in ['fl', 'fr', 'hl', 'hr']:
            s = math.sin(omega * t + phases[leg])
            if s > 0:
                lift = math.sin(math.pi * s)
                tg[i+1] = 0.0 + 0.6 * lift; tg[i+2] = -0.5 - 0.8 * lift
            else:
                push = s
                tg[i+1] = 0.4 - 0.2 * push + (self.cmd_vx * 0.2); tg[i+2] = -1.0 + 0.2 * push
            if leg in ['fl', 'hl']: tg[i] = self.cmd_wz * 0.3 + self.cmd_vy * 0.4
            else: tg[i] = -self.cmd_wz * 0.3 - self.cmd_vy * 0.4
            i += 3
            
        if self.model.nu >= 16:
            if self.is_grabbing:
                tg[14] = 0.5; tg[15] = -0.5
            else:
                tg[14] = 0.0; tg[15] = 0.0
                
        for j_id in range(self.model.nu):
            jid = self.model.actuator_trnid[j_id, 0]
            rng = self.model.jnt_range[jid]
            tg[j_id] = np.clip(tg[j_id], rng[0], rng[1])
        return tg

    def _process_api_commands(self):
        while not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            try:
                if cmd.get("cmd") == "set_target":
                    self.target_vx = cmd.get("vx", 0.0)
                    self.target_vy = cmd.get("vy", 0.0)
                    self.target_wz = cmd.get("wz", 0.0)
                elif cmd.get("cmd") == "reset":
                    self._reset_sim()
                elif cmd.get("cmd") == "get_state":
                    state = {
                        "pos": self.data.xpos[self.trunk_id].tolist(),
                        "vel": self.data.qvel[0:3].tolist(),
                        "battery": self.battery_percent
                    }
                    self.state_queue.put(state)
            except Exception as e:
                self.log(f"[ERROR] API cmd failed: {e}")

    def _pure_pursuit(self):
        if not self.path_pts or self.path_idx >= len(self.path_pts):
            self.clear_waypoints()
            self.target_vx = 0.0; self.target_wz = 0.0
            return

        robot_pos = self.data.xpos[self.trunk_id][:2]
        dists = np.linalg.norm(np.array(self.path_pts) - np.array(robot_pos), axis=1)
        closest_idx = np.argmin(dists)
        
        look_ahead = 0.4 
        look_idx = closest_idx
        acc_dist = 0.0
        while look_idx < len(self.path_pts) - 1 and acc_dist < look_ahead:
            acc_dist += np.linalg.norm(np.array(self.path_pts[look_idx+1]) - np.array(self.path_pts[look_idx]))
            look_idx += 1
            
        target_pt = self.path_pts[look_idx]
        
        if closest_idx == len(self.path_pts) - 1 and np.linalg.norm(np.array(target_pt) - np.array(robot_pos)) < 0.3:
            self.log("[NAV] Path completed!")
            self.clear_waypoints()
            return
            
        dx = target_pt[0] - robot_pos[0]
        dy = target_pt[1] - robot_pos[1]
        target_angle = math.atan2(dy, dx)
        angle_err = target_angle - self.robot_yaw
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi
        
        self.target_vx = 1.0
        self.target_wz = 2.0 * math.sin(angle_err) / look_ahead

    def _update_grabbing(self):
        if not self.is_grabbing:
            self.grabbed_bid = -1
            return
            
        if self.grabbed_bid == -1:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1 = self.model.geom_bodyid[c.geom1]
                g2 = self.model.geom_bodyid[c.geom2]
                
                b_gripper = g1 in [self.finger_l_id, self.finger_r_id, self.gripper_base_id]
                obj_is_gripper = g2 in [self.finger_l_id, self.finger_r_id, self.gripper_base_id]
                
                if b_gripper:
                    bid = g2
                elif obj_is_gripper:
                    bid = g1
                else:
                    continue
                    
                b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if b_name and "spawn_" in b_name:
                    self.grabbed_bid = bid
                    g_pos = self.data.xpos[self.gripper_base_id]
                    g_quat = self.data.xquat[self.gripper_base_id]
                    o_pos = self.data.xpos[bid]
                    o_quat = self.data.xquat[bid]
                    
                    diff = o_pos - g_pos
                    self.grab_offset_pos = mujoco.mju_rotVecQuat(diff, np.array([-g_quat[0], -g_quat[1], -g_quat[2], -g_quat[3]]))
                    self.grab_offset_quat = mujoco.mju_mulQuat(np.array([-g_quat[0], -g_quat[1], -g_quat[2], -g_quat[3]]), o_quat)
                    self.log(f"[INFO] Grabbed object: {b_name}")
                    break
        else:
            g_pos = self.data.xpos[self.gripper_base_id]
            g_quat = self.data.xquat[self.gripper_base_id]
            
            new_pos = g_pos + mujoco.mju_rotVecQuat(self.grab_offset_pos, g_quat)
            new_quat = mujoco.mju_mulQuat(g_quat, self.grab_offset_quat)
            
            jid = self.model.body_jntadr[self.grabbed_bid]
            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            
            self.data.qpos[qpos_adr:qpos_adr+3] = new_pos
            self.data.qpos[qpos_adr+3:qpos_adr+7] = new_quat
            self.data.qvel[qvel_adr:qvel_adr+6] = 0.0

    def _calculate_zmp(self):
        cop = np.zeros(2)
        total_f = 0.0
        self.support_poly_pts = []
        
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = self.model.geom_bodyid[c.geom1]
            g2 = self.model.geom_bodyid[c.geom2]
            
            b1_is_ground = g1 == 0 and ("ground" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "stair" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1))
            b2_is_ground = g2 == 0 and ("ground" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "stair" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
            
            if not (b1_is_ground or b2_is_ground):
                continue
                
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            fn = f[2] 
            if fn > 0.5:
                pos = c.pos
                cop[0] += pos[0] * fn
                cop[1] += pos[1] * fn
                total_f += fn
                
                foot_bid = g1 if b2_is_ground else g2
                if foot_bid in self.viewport.foot_body_ids:
                    self.support_poly_pts.append((pos[0], pos[1]))
                    
        if total_f > 1.0:
            self.cop_xy = cop / total_f
            self.cop_valid = True
        else:
            self.cop_valid = False
            
        if self.cop_valid and len(self.support_poly_pts) >= 3:
            def point_in_poly(pt, poly):
                x, y = pt
                n = len(poly)
                inside = False
                p1x, p1y = poly[0]
                for i in range(n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xints:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside
            self.zmp_inside = point_in_poly(self.cop_xy, self.support_poly_pts)
        elif self.cop_valid and len(self.support_poly_pts) > 0:
            min_dist = min(np.hypot(self.cop_xy[0]-p[0], self.cop_xy[1]-p[1]) for p in self.support_poly_pts)
            self.zmp_inside = min_dist < 0.05
        else:
            self.zmp_inside = False

    def _step_simulation(self, n_steps):
        sim_dt = float(self.model.opt.timestep)

        if self.nav_goal is not None and self.trunk_id >= 0:
            dx = self.nav_goal[0] - self.data.xpos[self.trunk_id][0]
            dy = self.nav_goal[1] - self.data.xpos[self.trunk_id][1]
            dist = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            angle_err = target_angle - self.robot_yaw
            angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi

            if dist > 0.3:
                front_right = self.lidar_ranges[1:30]
                front_left = self.lidar_ranges[90:119]
                min_right = np.min(front_right)
                min_left = np.min(front_left)
                
                if min_right < 0.6 and min_left < 0.6:
                    self.obstacle_blocked = True
                    self.target_vx = 0.0
                    self.target_wz = 1.0 if min_right < min_left else -1.0
                elif min_right < 0.6 or min_left < 0.6:
                    self.obstacle_blocked = True
                    self.target_vx = 0.3
                    diff = min_left - min_right
                    self.target_wz = np.clip(diff * 3.0, -1.0, 1.0)
                else:
                    self.obstacle_blocked = False
                    self.target_vx = 1.0
                    self.target_wz = np.clip(angle_err * 2.0, -1.0, 1.0)
            else:
                self.target_vx = 0.0; self.target_wz = 0.0
                self.cancel_nav()
                self.log("[NAV] Goal reached!")
        elif self.path_pts:
            self._pure_pursuit()
        else:
            self.obstacle_blocked = False

        for _ in range(n_steps):
            if self.kinematic_mode:
                self.data.ctrl[:] = 0
                self.data.qfrc_applied[:] = 0
                self.data.qvel[:] = 0
            elif self.recovering:
                self.data.ctrl[:] = 0
            elif self.battery_percent <= 0.0:
                self.data.ctrl[:] = 0
            else:
                alpha = min(1.0, sim_dt * 5.0)
                self.cmd_vx += (self.target_vx - self.cmd_vx) * alpha
                self.cmd_vy += (self.target_vy - self.cmd_vy) * alpha
                self.cmd_wz += (self.target_wz - self.cmd_wz) * alpha
                tgt = self._gait_targets(self.data.time) if self.gait_enabled else self.manual_targets
                self.data.ctrl[:] = tgt
                if self.trunk_id >= 0:
                    t = self.data.time
                    wind_x_eff = self.wind_x * (1.0 + 0.3 * math.sin(t * 2.5))
                    wind_y_eff = self.wind_y * (1.0 + 0.3 * math.cos(t * 1.7))
                    self.data.xfrc_applied[self.trunk_id, 0] = wind_x_eff
                    self.data.xfrc_applied[self.trunk_id, 1] = wind_y_eff
                    self.data.xfrc_applied[self.trunk_id, 2] = 0.0

            mujoco.mj_step(self.model, self.data)
            
            self._update_grabbing()

            if not self.kinematic_mode:
                instant_power = np.sum(np.maximum(0, 
                    self.data.actuator_force * self.data.qvel[self.act_qvel]
                ))
                total_power = 2.0 + instant_power 
                self.battery_percent = max(0.0, self.battery_percent - total_power * sim_dt * 4e-4)

            self.trail_step += 1
            if self.trail_step % 10 == 0 and self.trunk_id >= 0:
                mid = self.trail_mocap_ids[self.trail_idx]
                if mid >= 0:
                    self.data.mocap_pos[mid] = self.data.xpos[self.trunk_id]
                self.trail_idx = (self.trail_idx + 1) % 50

        self.sim_time = self.data.time
        self._calculate_zmp()

    def _tick(self):
        now = self._clock.elapsed() / 1000.0
        real_dt = now - self._last_real_t
        self._last_real_t = now
        real_dt = min(real_dt, 0.1)
        
        if self.isMinimized():
            return

        self._process_api_commands()

        if self.sim_running and not self.is_scrubbing:
            self._running_real_t += real_dt
            sim_dt = float(self.model.opt.timestep)
            steps = max(1, min(int(real_dt * self.sim_speed / sim_dt), 80))
            self._step_simulation(steps)
            
            self.timeline_hist.append((self.data.qpos.copy(), self.data.qvel.copy()))
            self.timeline_slider.setMaximum(max(0, len(self.timeline_hist)-1))
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(len(self.timeline_hist)-1)
            self.timeline_slider.blockSignals(False)

        self._fps_count += 1; self._fps_acc += real_dt
        if self._fps_acc >= 0.5:
            self._fps = self._fps_count / self._fps_acc; self._fps_count = 0; self._fps_acc = 0.0

        self.lidar_ray_step += 1
        if self.lidar_ray_step >= 3:
            self.cast_lidar()
            self.lidar_ray_step = 0

        self._coll_update_timer += real_dt
        if self._coll_update_timer > 0.2:
            self._update_collision_tree()
            self._coll_update_timer = 0.0

        self._update_state(real_dt)

        if self.chase_cam:
            if self.trunk_id >= 0:
                self.viewport.cam_main.lookat[0] = self.data.xpos[self.trunk_id][0]
                self.viewport.cam_main.lookat[1] = self.data.xpos[self.trunk_id][1]
                self.viewport.cam_main.lookat[2] = self.data.xpos[self.trunk_id][2]
                target_az = math.degrees(self.robot_yaw)
                diff = (target_az - self.viewport.cam_main.azimuth + 180) % 360 - 180
                self.viewport.cam_main.azimuth += diff * 0.1

        self.viewport.render_scene()

        if self.recording and self.viewport._image:
            if now - self._last_rec_t >= 1.0 / self.rec_target_fps:
                path = os.path.join(self.rec_folder, f"frame_{self.rec_frame:04d}.png")
                self.viewport._image.save(path); self.rec_frame += 1
                self._last_rec_t = now

        self.torque_plot.push_data(self.data.actuator_force)
        self.lbl_time.setText(f"t = {self.sim_time:.3f} s")
        self.lbl_fps.setText(f"FPS: {self._fps:.0f}")
        self.lbl_ncon.setText(f"contacts: {self.data.ncon}")
        
        if self.sim_running:
            rtf = self.sim_time / self._running_real_t if self._running_real_t > 1e-6 else 0.0
            self.lbl_rtf.setText(f"RTF: {rtf:.2f}x")
        else:
            self.lbl_rtf.setText("RTF: — (paused)")

    def _update_state(self, real_dt):
        if self.trunk_id < 0: return
        pos = self.data.xpos[self.trunk_id]; quat = self.data.xquat[self.trunk_id]

        speed = 0.0
        if self.vel_sid >= 0:
            adr = self.model.sensor_adr[self.vel_sid]
            vx = self.data.sensordata[adr]; vy = self.data.sensordata[adr+1]
            speed = math.hypot(vx, vy)

        w, x, y, z = quat
        sinr = 2.0 * (w * x + y * z); cosr = 1.0 - 2.0 * (x * x + y * y); roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x); pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (w * z + x * y); cosy = 1.0 - 2.0 * (y * y + z * z); self.robot_yaw = math.atan2(siny, cosy)

        if self.recovering:
            self.recover_timer -= real_dt
            if self.recover_timer <= 0:
                self.recovering = False
                self.log("[INFO] Recovery complete.")
        elif not self.kinematic_mode:
            if abs(roll) > 1.2 or abs(pitch) > 1.2:
                self.recovering = True
                self.recover_timer = 2.0
                self.log("[WARNING] Robot tipped over! Auto-recovering to Home stance...")
                self.reset_robot_only()
                self.apply_default_pose("home")

        self.lbl_pos.setText(f"{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}")
        self.lbl_vel.setText(f"{speed:.2f}")
        self.lbl_battery.setText(f"{self.battery_percent:.0f}%")
        if self.battery_percent < 20:
            self.lbl_battery.setStyleSheet("color: #ff5555; font-family: Consolas; font-weight: bold;")
            if self.battery_percent <= 0.0 and not self._battery_dead_logged:
                self.log("[CRITICAL] Battery depleted! Robot controls disabled.")
                self._battery_dead_logged = True
        elif self.battery_percent < 50:
            self.lbl_battery.setStyleSheet("color: #ffaa00; font-family: Consolas;")
        else:
            self.lbl_battery.setStyleSheet("color: #55ff55; font-family: Consolas;")

        if self.gyro_sid >= 0:
            adr = self.model.sensor_adr[self.gyro_sid]
            gx = self.data.sensordata[adr]; gy = self.data.sensordata[adr+1]; gz = self.data.sensordata[adr+2]
            if self.imu_noise > 0:
                gx += np.random.normal(0, self.imu_noise)
                gy += np.random.normal(0, self.imu_noise)
                gz += np.random.normal(0, self.imu_noise)
            self.imu_graph.push_data(gx, gy, gz)
        else:
            self.imu_graph.push_data(0, 0, 0)
            
        com_x = self.data.subtree_com[self.trunk_id][0]
        com_y = self.data.subtree_com[self.trunk_id][1]
        self.com_plot.push_data(com_x, com_y)

        items = self.tree.selectedItems()
        if items and items[0].text(1) == "body":
            bid = items[0].data(0, Qt.UserRole)
            if bid is not None:
                p = self.data.xpos[bid]; v = self.data.cvel[bid][3:6]; q = self.data.xquat[bid]
                self.lbl_sel_pos.setText(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
                self.lbl_sel_quat.setText(f"{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}")
                self.lbl_sel_vel.setText(f"{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}")

    def keyPressEvent(self, ev):
        if ev.isAutoRepeat():
            super().keyPressEvent(ev); return
        k = ev.key()
        if k == Qt.Key_Escape:
            if self.viewport.nav_goal_mode:
                self.viewport.nav_goal_mode = False
                self.viewport.setCursor(Qt.ArrowCursor)
                self.log("[INFO] Navigation goal cancelled.")
            else:
                self.close()
        elif k == Qt.Key_Space: self._toggle_play()
        elif k == Qt.Key_R and (ev.modifiers() & Qt.ControlModifier):
            if (ev.modifiers() & Qt.ShiftModifier): self.reset_robot_only()
            else: self._reset_sim()
        elif k == Qt.Key_Z and (ev.modifiers() & Qt.ControlModifier):
            self.undo()
        elif k == Qt.Key_Y and (ev.modifiers() & Qt.ControlModifier):
            self.redo()
        elif k == Qt.Key_C: self.viewport._reset_cam()
        elif k == Qt.Key_K: 
            self.a_kinematic.setChecked(not self.a_kinematic.isChecked())
            self._toggle_kinematic(self.a_kinematic.isChecked())
        elif k == Qt.Key_G:
            self.btn_grab.setChecked(not self.btn_grab.isChecked())
        elif k == Qt.Key_W: self.target_vx = 1.0
        elif k == Qt.Key_S: self.target_vx = -1.0
        elif k == Qt.Key_A: self.target_wz = 1.0
        elif k == Qt.Key_D: self.target_wz = -1.0
        elif k == Qt.Key_Q: self.target_vy = 1.0
        elif k == Qt.Key_E: self.target_vy = -1.0
        else: super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.isAutoRepeat():
            super().keyReleaseEvent(ev); return
        k = ev.key()
        if k in [Qt.Key_W, Qt.Key_S]: self.target_vx = 0.0
        elif k in [Qt.Key_A, Qt.Key_D]: self.target_wz = 0.0
        elif k in [Qt.Key_Q, Qt.Key_E]: self.target_vy = 0.0
        else: super().keyReleaseEvent(ev)

    def _restore_settings(self):
        s = QSettings("PoseStudio", "MuJoCo")
        self.restoreGeometry(s.value("geometry", b""))
        self.restoreState(s.value("windowState", b""))

    def closeEvent(self, ev):
        self.timer.stop()
        if self.api_server:
            self.api_server.stop()
        if self.viewport._renderer:
            self.viewport._renderer.close()
        if self.viewport._pip_renderer:
            self.viewport._pip_renderer.close()
            
        s = QSettings("PoseStudio", "MuJoCo")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        super().closeEvent(ev)


# ============================================================================
# MJCF Model & Environments
# ============================================================================
ROBOT_XML = r"""
<body name="trunk" pos="0 0 0.45">
  <freejoint name="root"/>
  <geom name="trunk" type="box" size="0.22 0.11 0.05" material="body_mat" mass="2.5"/>
  <site name="imu_site" pos="0 0 0" size="0.01" rgba="0 0 1 0.5"/>
  <site name="height_site" pos="0 0 -0.05" size="0.01" rgba="0 1 0 0.5"/>
  <body name="neck_base" pos="0.18 0 0.04">
    <joint name="neck_yaw" axis="0 0 1" range="-1.5 1.5"/>
    <geom type="cylinder" size="0.03 0.04" rgba="0.8 0.8 0.8 1"/>
    <body name="head" pos="0 0 0.06">
      <joint name="neck_pitch" axis="1 0 0" range="-0.5 0.5"/>
      <geom type="box" size="0.05 0.06 0.03" material="head_mat"/>
      <geom type="cylinder" pos="0.04 0 0" zaxis="1 0 0" size="0.015 0.01" rgba="0.9 0.9 0.2 1"/>
      <camera name="robot_cam" pos="0.06 0 0" fovy="70"/>
      <site name="lidar_site" pos="0.06 0 0.02" size="0.01" rgba="1 0 0 1"/>
      
      <body name="gripper_base" pos="0.08 0 0">
        <geom type="box" size="0.02 0.02 0.01" rgba="0.2 0.2 0.2 1"/>
        <body name="finger_l" pos="0.02 0.02 0">
          <joint name="finger_l_jnt" axis="0 0 1" range="0 0.5"/>
          <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.005" rgba="0.8 0.8 0.2 1"/>
        </body>
        <body name="finger_r" pos="0.02 -0.02 0">
          <joint name="finger_r_jnt" axis="0 0 1" range="-0.5 0"/>
          <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.005" rgba="0.8 0.8 0.2 1"/>
        </body>
      </body>
    </body>
  </body>
  <body name="fl_hip" pos="0.18 0.09 0">
    <joint name="fl_haa" axis="0 0 1" range="-0.6 0.6"/>
    <geom class="leg" fromto="0 0 0 0 0.05 0"/>
    <body name="fl_thigh" pos="0 0.05 0">
      <joint name="fl_hfe" axis="1 0 0" range="-1.5 1.5"/>
      <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
      <body name="fl_calf" pos="0 0 -0.14">
        <joint name="fl_kfe" axis="1 0 0" range="-2.5 0.0"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
        <geom name="fl_foot" type="cylinder" pos="0 0 -0.14" size="0.025 0.01" rgba="0.1 0.1 0.1 1" density="1200"/>
        <site name="fl_foot_site" pos="0 0 -0.15" size="0.02" rgba="1 0 0 0.3"/>
      </body>
    </body>
  </body>
  <body name="fr_hip" pos="0.18 -0.09 0">
    <joint name="fr_haa" axis="0 0 1" range="-0.6 0.6"/>
    <geom class="leg" fromto="0 0 0 0 -0.05 0"/>
    <body name="fr_thigh" pos="0 -0.05 0">
      <joint name="fr_hfe" axis="1 0 0" range="-1.5 1.5"/>
      <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
      <body name="fr_calf" pos="0 0 -0.14">
        <joint name="fr_kfe" axis="1 0 0" range="-2.5 0.0"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
        <geom name="fr_foot" type="cylinder" pos="0 0 -0.14" size="0.025 0.01" rgba="0.1 0.1 0.1 1" density="1200"/>
        <site name="fr_foot_site" pos="0 0 -0.15" size="0.02" rgba="1 0 0 0.3"/>
      </body>
    </body>
  </body>
  <body name="hl_hip" pos="-0.18 0.09 0">
    <joint name="hl_haa" axis="0 0 1" range="-0.6 0.6"/>
    <geom class="leg" fromto="0 0 0 0 0.05 0"/>
    <body name="hl_thigh" pos="0 0.05 0">
      <joint name="hl_hfe" axis="1 0 0" range="-1.5 1.5"/>
      <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
      <body name="hl_calf" pos="0 0 -0.14">
        <joint name="hl_kfe" axis="1 0 0" range="-2.5 0.0"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
        <geom name="hl_foot" type="cylinder" pos="0 0 -0.14" size="0.025 0.01" rgba="0.1 0.1 0.1 1" density="1200"/>
        <site name="hl_foot_site" pos="0 0 -0.15" size="0.02" rgba="1 0 0 0.3"/>
      </body>
    </body>
  </body>
  <body name="hr_hip" pos="-0.18 -0.09 0">
    <joint name="hr_haa" axis="0 0 1" range="-0.6 0.6"/>
    <geom class="leg" fromto="0 0 0 0 -0.05 0"/>
    <body name="hr_thigh" pos="0 -0.05 0">
      <joint name="hr_hfe" axis="1 0 0" range="-1.5 1.5"/>
      <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
      <body name="hr_calf" pos="0 0 -0.14">
        <joint name="hr_kfe" axis="1 0 0" range="-2.5 0.0"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.14"/>
        <geom name="hr_foot" type="cylinder" pos="0 0 -0.14" size="0.025 0.01" rgba="0.1 0.1 0.1 1" density="1200"/>
        <site name="hr_foot_site" pos="0 0 -0.15" size="0.02" rgba="1 0 0 0.3"/>
      </body>
    </body>
  </body>
</body>
"""

ACTUATOR_XML = r"""
<actuator>
  <general name="neck_yaw" joint="neck_yaw" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="neck_pitch" joint="neck_pitch" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fl_haa" joint="fl_haa" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fl_hfe" joint="fl_hfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fl_kfe" joint="fl_kfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fr_haa" joint="fr_haa" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fr_hfe" joint="fr_hfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="fr_kfe" joint="fr_kfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hl_haa" joint="hl_haa" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hl_hfe" joint="hl_hfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hl_kfe" joint="hl_kfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hr_haa" joint="hr_haa" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hr_hfe" joint="hr_hfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="hr_kfe" joint="hr_kfe" gainprm="150" biasprm="0 -150 -5" dyntype="filter" dynprm="0.02"/>
  <general name="finger_l" joint="finger_l_jnt" gainprm="50" biasprm="0 -50 0" ctrlrange="0 0.5"/>
  <general name="finger_r" joint="finger_r_jnt" gainprm="50" biasprm="0 -50 0" ctrlrange="-0.5 0"/>
</actuator>
"""

ASSETS_XML = r"""
<asset>
  <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.25 0.35" rgb2="0.05 0.10 0.15" width="512" height="512"/>
  <material name="grid_mat" texture="grid" texrepeat="20 20" reflectance="0.2"/>
  <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.7 1.0" rgb2="0.02 0.02 0.05" width="256" height="256"/>
  <texture name="body_tex" type="cube" builtin="gradient" rgb1="0.65 0.65 0.75" rgb2="0.15 0.15 0.25" width="128" height="128"/>
  <material name="body_mat" texture="body_tex" shininess="0.8" reflectance="0.5"/>
  <material name="head_mat" rgba="0.2 0.2 0.25 1" shininess="0.9" reflectance="0.6"/>
</asset>
"""

DEFAULTS_XML = r"""
<default>
  <joint damping="0.5" armature="0.02" frictionloss="0.05" limited="true" solreflimit="0.02 1" solimplimit="0.9 0.95 0.001"/>
  <geom condim="6" friction="1.0 0.1 0.05" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  <general ctrllimited="true" ctrlrange="-3.14 3.14" forcelimited="true" forcerange="-40 40"
           gaintype="fixed" biastype="affine"/>
  <default class="leg"><geom type="capsule" size="0.022" rgba="0.85 0.65 0.35 1" density="800"/></default>
</default>
"""

CONTACTS_XML = r"""
<contact>
  <exclude body1="trunk" body2="fl_hip"/>
  <exclude body1="trunk" body2="fr_hip"/>
  <exclude body1="trunk" body2="hl_hip"/>
  <exclude body1="trunk" body2="hr_hip"/>
  <exclude body1="fl_thigh" body2="fl_calf"/>
  <exclude body1="fr_thigh" body2="fr_calf"/>
  <exclude body1="hl_thigh" body2="hl_calf"/>
  <exclude body1="hr_thigh" body2="hr_calf"/>
</contact>
"""

TRAIL_XML = "".join([
    f'<body name="trail_{i}" pos="10 10 10" mocap="true">'
    f'<geom type="sphere" size="0.025" rgba="0 1 0 0.6" group="2"/></body>\n'
    for i in range(50)
])

SPAWN_XML = "".join([
    f'<body name="spawn_box_{i}" pos="10 10 10" mocap="true">'
    f'<freejoint name="spawn_box_{i}_jnt"/>'
    f'<geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.2 1" mass="0.5"/></body>\n'
    for i in range(3)
] + [
    f'<body name="spawn_sphere_{i}" pos="10 10 10" mocap="true">'
    f'<freejoint name="spawn_sph_{i}_jnt"/>'
    f'<geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" mass="0.5"/></body>\n'
    for i in range(3)
])

SENSORS_XML = r"""
<sensor>
  <gyro name="gyro" site="imu_site"/>
  <accelerometer name="accel" site="imu_site"/>
  <touch name="fl_touch" site="fl_foot_site"/>
  <touch name="fr_touch" site="fr_foot_site"/>
  <touch name="hl_touch" site="hl_foot_site"/>
  <touch name="hr_touch" site="hr_foot_site"/>
  <framelinvel name="base_vel" objtype="body" objname="trunk"/>
  <rangefinder name="height_sensor" site="height_site"/>
</sensor>
"""

ENV_FLAT = f"""
<mujoco model="env_flat">
  <size njmax="500" nconmax="200" nstack="5000000"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" iterations="50" tolerance="1e-8" density="1.2" viscosity="0.0002">
    <flag energy="enable" fwdinv="enable"/>
  </option>
  <compiler angle="radian" coordinate="local" autolimits="true" inertiafromgeom="auto" fusestatic="true"/>
  {DEFAULTS_XML} {ASSETS_XML}
  <worldbody>
    <light directional="true" diffuse="0.9 0.9 0.9" castshadow="true" pos="3 -3 5" dir="-1 1 -2"/>
    <light pos="0 0 6" diffuse="0.3 0.3 0.3"/>
    <geom name="ground" type="plane" size="20 20 0.1" material="grid_mat" group="1"/>
    <geom name="stair1" type="box" pos="1.5 0.0 0.05" size="0.5 0.5 0.05" rgba="0.8 0.8 0.2 0.8" group="1"/>
    <geom name="stair2" type="box" pos="2.5 0.0 0.10" size="0.5 0.5 0.10" rgba="0.8 0.8 0.2 0.8" group="1"/>
    <geom name="stair3" type="box" pos="3.5 0.0 0.15" size="0.5 0.5 0.15" rgba="0.8 0.8 0.2 0.8" group="1"/>
    <geom name="wall1" type="box" pos="-1.5 1.0 0.5" size="0.1 1.0 0.5" rgba="0.8 0.2 0.8 0.8" group="1"/>
    <geom name="wall2" type="box" pos="-1.5 -1.0 0.5" size="0.1 1.0 0.5" rgba="0.8 0.2 0.8 0.8" group="1"/>
    {ROBOT_XML}
  </worldbody>
  {CONTACTS_XML}
  {ACTUATOR_XML}
  {SENSORS_XML}
</mujoco>
"""

ENV_FLAT = ENV_FLAT.replace(
    "</worldbody>",
    TRAIL_XML + '<body name="goal_marker" pos="10 10 10" mocap="true">'
    '<geom type="sphere" size="0.1" rgba="1 0.2 0.2 0.8" group="2"/></body>\n'
    + SPAWN_XML +
    '\n</worldbody>'
)


# ============================================================================
# Stylesheet
# ============================================================================
STYLESHEET = """
QMainWindow { background: #1e1e1e; }
QDockWidget { border: 1px solid #333; padding: 0px; }
QDockWidget::title { text-align: left; background: #252526; padding: 6px 10px; color: #cccccc; font-weight: bold; border-bottom: 1px solid #333; }
QGroupBox { border: 1px solid #333; border-radius: 4px; margin-top: 12px; padding-top: 10px; font-weight: bold; color: #999; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; background: #1e1e1e; }
QPushButton { background: #3a3d41; border: 1px solid #3a3d41; padding: 6px 12px; border-radius: 3px; color: #ddd; }
QPushButton:hover { background: #4c5052; border: 1px solid #007acc; }
QPushButton:pressed { background: #2b2b2b; }
QPushButton#PlayBtn { background: #2d5a2d; font-weight: bold; }
QPushButton#PlayBtn:hover { background: #3a7a3a; }
QPushButton#RecBtn { background: #4a1a1a; font-weight: bold; }
QPushButton#RecBtn:hover { background: #6a2a2a; }
QToolBar { background: #2d2d2d; border-bottom: 1px solid #1a1a1a; padding: 4px; spacing: 4px; }
QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }
QTabBar::tab { background: #252526; color: #888; padding: 6px 12px; border: 1px solid #333; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }
QTabBar::tab:selected { background: #1e1e1e; color: #fff; border-bottom: 2px solid #007acc; }
QTreeWidget { background: #252526; border: 1px solid #333; color: #d4d4d4; alternate-background-color: #2d2d2d; }
QTreeWidget::item:selected { background: #094771; color: #ffffff; }
QTextEdit, QPlainTextEdit { background: #1e1e1e; border: 1px solid #333; color: #d4d4d4; }
QListWidget { background: #252526; border: 1px solid #333; color: #d4d4d4; }
QListWidget::item:selected { background: #094771; color: #ffffff; }
QScrollBar:vertical { border: none; background: #252526; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4c5052; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #007acc; }
QSlider::groove:horizontal { border: 1px solid #333; height: 4px; background: #252526; border-radius: 2px; }
QSlider::handle:horizontal { background: #007acc; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #1c97ea; }
QStatusBar { background: #007acc; border-top: 1px solid #1a1a1a; color: #ffffff; font-weight: bold; }
QStatusBar::item { border: none; }
QStatusBar QLabel { margin-left: 10px; margin-right: 10px; color: #ffffff; }
QLabel { color: #ccc; }
QMenu { background: #252526; border: 1px solid #333; }
QMenu::item { padding: 5px 20px; }
QMenu::item:selected { background: #094771; }
QLineEdit { background: #252526; border: 1px solid #333; color: #d4d4d4; padding: 4px; }
QComboBox { background: #3a3d41; border: 1px solid #3a3d41; padding: 4px; color: #ddd; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #252526; border: 1px solid #333; color: #d4d4d4; selection-background-color: #094771; }
QDoubleSpinBox { background: #252526; border: 1px solid #333; color: #d4d4d4; padding: 2px; }
"""

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = RobotSimWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
