import os

# ============================================================
# LEGACY KERAS
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "traffic_classifier.h5"

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    print("Model loaded successfully!")

except Exception as e:
    print("Error loading model:")
    print(e)
    raise


# ============================================================
# TRAFFIC SIGN CLASSES
# ============================================================

classes = {
    1: "Speed limit (20km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (50km/h)",
    4: "Speed limit (60km/h)",
    5: "Speed limit (70km/h)",
    6: "Speed limit (80km/h)",
    7: "End of speed limit (80km/h)",
    8: "Speed limit (100km/h)",
    9: "Speed limit (120km/h)",
    10: "No passing",
    11: "No passing veh over 3.5 tons",
    12: "Right-of-way at intersection",
    13: "Priority road",
    14: "Yield",
    15: "Stop",
    16: "No vehicles",
    17: "Veh > 3.5 tons prohibited",
    18: "No entry",
    19: "General caution",
    20: "Dangerous curve left",
    21: "Dangerous curve right",
    22: "Double curve",
    23: "Bumpy road",
    24: "Slippery road",
    25: "Road narrows on the right",
    26: "Road work",
    27: "Traffic signals",
    28: "Pedestrians",
    29: "Children crossing",
    30: "Bicycles crossing",
    31: "Beware of ice/snow",
    32: "Wild animals crossing",
    33: "End speed + passing limits",
    34: "Turn right ahead",
    35: "Turn left ahead",
    36: "Ahead only",
    37: "Go straight or right",
    38: "Go straight or left",
    39: "Keep right",
    40: "Keep left",
    41: "Roundabout mandatory",
    42: "End of no passing",
    43: "End no passing veh > 3.5 tons"
}


# ============================================================
# DESIGN COLORS
# ============================================================

BG_TOP = "#050915"
BG_BOTTOM = "#0A1224"

PANEL = "#10192B"
PANEL_TOP = "#17243D"
PANEL_DARK = "#091121"
PANEL_INNER = "#0B1527"

CYAN = "#00E8D3"
CYAN_BRIGHT = "#7AFFF4"
BLUE = "#20AFFF"
BLUE_DARK = "#0878C9"

WHITE = "#F4FAFF"
LIGHT = "#C7D7EA"
GREY = "#7588A6"
GREY_DARK = "#40516D"

GREEN = "#00E79C"
RED = "#FF5268"
YELLOW = "#FFD166"

BORDER = "#263B5C"
BORDER_LIGHT = "#35577F"

SHADOW = "#01040B"
GRID = "#101D32"


# ============================================================
# FONTS
# ============================================================

TITLE_FONT = ("Segoe UI", 31, "bold")
SUBTITLE_FONT = ("Segoe UI", 11, "bold")
CARD_TITLE_FONT = ("Segoe UI", 12, "bold")
RESULT_FONT = ("Segoe UI", 21, "bold")
SMALL_FONT = ("Segoe UI", 9)
TINY_FONT = ("Segoe UI", 8)
BUTTON_FONT = ("Segoe UI", 11, "bold")
DIGITAL_FONT = ("Consolas", 10, "bold")


# ============================================================
# WINDOW
# ============================================================

top = tk.Tk()
top.title("Traffic Sign Classifier — AI Vision Console")
top.geometry("1440x900")
top.minsize(1180, 760)
top.configure(bg=BG_TOP)
top.resizable(True, True)

try:
    top.iconbitmap("traffic.ico")
except Exception:
    pass


# ============================================================
# MASTER CANVAS
# ============================================================

canvas = tk.Canvas(
    top,
    bg=BG_TOP,
    highlightthickness=0,
    bd=0
)

canvas.pack(fill="both", expand=True)


# ============================================================
# LAYOUT CONSTANTS
# ============================================================

DESIGN_WIDTH = 1440
DESIGN_HEIGHT = 900

LEFT_X1 = 60
LEFT_Y1 = 158
LEFT_X2 = 920
LEFT_Y2 = 715

RIGHT_X1 = 950
RIGHT_Y1 = 158
RIGHT_X2 = 1380
RIGHT_Y2 = 715

IMAGE_X1 = LEFT_X1 + 32
IMAGE_Y1 = LEFT_Y1 + 104
IMAGE_X2 = LEFT_X2 - 32
IMAGE_Y2 = LEFT_Y2 - 34

STATUS_X1 = RIGHT_X1 + 28
STATUS_Y1 = RIGHT_Y1 + 102
STATUS_X2 = RIGHT_X2 - 28
STATUS_Y2 = STATUS_Y1 + 82

RESULT_X1 = RIGHT_X1 + 28
RESULT_Y1 = RIGHT_Y1 + 210
RESULT_X2 = RIGHT_X2 - 28
RESULT_Y2 = RESULT_Y1 + 218

BUTTON_Y = 766


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hex_to_rgb(color):
    color = color.replace("#", "")
    return tuple(
        int(color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(rgb[0]))),
        max(0, min(255, int(rgb[1]))),
        max(0, min(255, int(rgb[2])))
    )


def blend_colors(color1, color2, ratio):
    rgb1 = np.array(hex_to_rgb(color1), dtype=float)
    rgb2 = np.array(hex_to_rgb(color2), dtype=float)
    return rgb_to_hex(rgb1 + (rgb2 - rgb1) * ratio)


def rounded_box(
    x1,
    y1,
    x2,
    y2,
    radius,
    fill,
    outline="",
    width=1,
    tags=()
):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1
    ]

    return canvas.create_polygon(
        points,
        smooth=True,
        splinesteps=24,
        fill=fill,
        outline=outline,
        width=width,
        tags=tags
    )


def draw_background(width, height):
    canvas.delete("background")

    width = max(width, DESIGN_WIDTH)
    height = max(height, DESIGN_HEIGHT)

    # High-resolution vertical gradient.
    bands = 150

    for index in range(bands):
        ratio = index / max(bands - 1, 1)
        color = blend_colors(BG_TOP, BG_BOTTOM, ratio)
        y1 = int((height / bands) * index)
        y2 = int((height / bands) * (index + 1)) + 1

        canvas.create_rectangle(
            0,
            y1,
            width,
            y2,
            fill=color,
            outline="",
            tags="background"
        )

    # Perspective-style technical grid.
    for x in range(0, width + 1, 64):
        canvas.create_line(
            x,
            128,
            x,
            height,
            fill=GRID,
            width=1,
            tags="background"
        )

    for y in range(128, height + 1, 48):
        canvas.create_line(
            0,
            y,
            width,
            y,
            fill=GRID,
            width=1,
            tags="background"
        )

    # Horizon illumination.
    canvas.create_line(
        0,
        138,
        width,
        138,
        fill="#153A55",
        width=1,
        tags="background"
    )

    canvas.create_line(
        0,
        140,
        width,
        140,
        fill="#0A1C30",
        width=1,
        tags="background"
    )

    # Corner technical decorations.
    canvas.create_line(
        22, 112,
        22, 42,
        92, 42,
        fill=BORDER,
        width=2,
        tags="background"
    )

    canvas.create_line(
        width - 22, 112,
        width - 22, 42,
        width - 92, 42,
        fill=BORDER,
        width=2,
        tags="background"
    )

    canvas.create_line(
        22, height - 72,
        22, height - 24,
        92, height - 24,
        fill=BORDER,
        width=2,
        tags="background"
    )

    canvas.create_line(
        width - 22, height - 72,
        width - 22, height - 24,
        width - 92, height - 24,
        fill=BORDER,
        width=2,
        tags="background"
    )

    canvas.tag_lower("background")


def draw_3d_panel(x1, y1, x2, y2, radius=22):
    # Multiple shadows create depth without extra libraries.
    rounded_box(
        x1 + 13,
        y1 + 17,
        x2 + 13,
        y2 + 17,
        radius,
        "#01030A"
    )

    rounded_box(
        x1 + 8,
        y1 + 11,
        x2 + 8,
        y2 + 11,
        radius,
        "#030711"
    )

    panel_id = rounded_box(
        x1,
        y1,
        x2,
        y2,
        radius,
        PANEL,
        BORDER,
        1
    )

    # Upper highlight gives the card a beveled surface.
    canvas.create_line(
        x1 + radius,
        y1 + 2,
        x2 - radius,
        y1 + 2,
        fill=BORDER_LIGHT,
        width=1
    )

    # Left reflection.
    canvas.create_line(
        x1 + 2,
        y1 + radius,
        x1 + 2,
        y2 - radius,
        fill="#172B47",
        width=1
    )

    return panel_id


def draw_section_title(x, y, title, subtitle, accent):
    canvas.create_text(
        x,
        y,
        text=title,
        anchor="w",
        fill=WHITE,
        font=CARD_TITLE_FONT
    )

    canvas.create_text(
        x,
        y + 25,
        text=subtitle,
        anchor="w",
        fill=GREY,
        font=SMALL_FONT
    )

    canvas.create_rectangle(
        x,
        y + 45,
        x + 112,
        y + 48,
        fill=accent,
        outline=""
    )

    canvas.create_rectangle(
        x + 112,
        y + 45,
        x + 176,
        y + 48,
        fill=BORDER,
        outline=""
    )


def draw_corner_brackets(x1, y1, x2, y2, color):
    length = 28
    offset = 13

    canvas.create_line(
        x1 + offset, y1 + offset + length,
        x1 + offset, y1 + offset,
        x1 + offset + length, y1 + offset,
        fill=color,
        width=2
    )

    canvas.create_line(
        x2 - offset - length, y1 + offset,
        x2 - offset, y1 + offset,
        x2 - offset, y1 + offset + length,
        fill=color,
        width=2
    )

    canvas.create_line(
        x1 + offset, y2 - offset - length,
        x1 + offset, y2 - offset,
        x1 + offset + length, y2 - offset,
        fill=color,
        width=2
    )

    canvas.create_line(
        x2 - offset - length, y2 - offset,
        x2 - offset, y2 - offset,
        x2 - offset, y2 - offset - length,
        fill=color,
        width=2
    )


# ============================================================
# BACKGROUND
# ============================================================

draw_background(DESIGN_WIDTH, DESIGN_HEIGHT)


# ============================================================
# HEADER
# ============================================================

canvas.create_text(
    62,
    48,
    text="TSC // VISION SYSTEM",
    anchor="w",
    fill=GREY,
    font=DIGITAL_FONT
)

canvas.create_text(
    62,
    72,
    text="NEURAL ROAD INTELLIGENCE",
    anchor="w",
    fill=GREY_DARK,
    font=TINY_FONT
)

header_title_glow = canvas.create_text(
    DESIGN_WIDTH // 2 + 2,
    66,
    text="TRAFFIC SIGN CLASSIFIER",
    fill="#063A42",
    font=TITLE_FONT
)

header_title = canvas.create_text(
    DESIGN_WIDTH // 2,
    63,
    text="TRAFFIC SIGN CLASSIFIER",
    fill=WHITE,
    font=TITLE_FONT
)

canvas.create_text(
    DESIGN_WIDTH // 2,
    104,
    text="AI-POWERED ROAD SIGN RECOGNITION SYSTEM",
    fill=CYAN,
    font=SUBTITLE_FONT
)

canvas.create_text(
    DESIGN_WIDTH // 2,
    127,
    text="UPLOAD  /  ANALYZE  /  IDENTIFY",
    fill=GREY,
    font=TINY_FONT
)

# Header status block.
canvas.create_rectangle(
    1197,
    43,
    1378,
    103,
    fill=PANEL_DARK,
    outline=BORDER,
    width=1
)

system_halo = canvas.create_oval(
    1212,
    59,
    1234,
    81,
    fill="",
    outline="#0B684E",
    width=1
)

system_dot = canvas.create_oval(
    1218,
    65,
    1228,
    75,
    fill=GREEN,
    outline=""
)

canvas.create_text(
    1241,
    64,
    text="SYSTEM ONLINE",
    anchor="w",
    fill=GREEN,
    font=("Segoe UI", 9, "bold")
)

clock_text = canvas.create_text(
    1241,
    84,
    text="00:00:00",
    anchor="w",
    fill=LIGHT,
    font=DIGITAL_FONT
)


# ============================================================
# MAIN 3-D PANELS
# ============================================================

draw_3d_panel(
    LEFT_X1,
    LEFT_Y1,
    LEFT_X2,
    LEFT_Y2
)

draw_3d_panel(
    RIGHT_X1,
    RIGHT_Y1,
    RIGHT_X2,
    RIGHT_Y2
)

draw_section_title(
    LEFT_X1 + 32,
    LEFT_Y1 + 31,
    "IMAGE ANALYSIS",
    "High-resolution traffic sign preview",
    CYAN
)

draw_section_title(
    RIGHT_X1 + 28,
    RIGHT_Y1 + 31,
    "AI PREDICTION",
    "CNN classification output",
    BLUE
)


# ============================================================
# IMAGE VIEWPORT
# ============================================================

rounded_box(
    IMAGE_X1 + 9,
    IMAGE_Y1 + 12,
    IMAGE_X2 + 9,
    IMAGE_Y2 + 12,
    17,
    SHADOW
)

rounded_box(
    IMAGE_X1,
    IMAGE_Y1,
    IMAGE_X2,
    IMAGE_Y2,
    17,
    PANEL_INNER,
    BORDER_LIGHT,
    1
)

# Layered inner edge.
rounded_box(
    IMAGE_X1 + 6,
    IMAGE_Y1 + 6,
    IMAGE_X2 - 6,
    IMAGE_Y2 - 6,
    14,
    PANEL_INNER,
    "#142844",
    1
)

draw_corner_brackets(
    IMAGE_X1,
    IMAGE_Y1,
    IMAGE_X2,
    IMAGE_Y2,
    CYAN
)

# Image viewport information.
canvas.create_text(
    IMAGE_X1 + 24,
    IMAGE_Y1 + 22,
    text="INPUT CHANNEL 01",
    anchor="w",
    fill=GREY_DARK,
    font=TINY_FONT
)

resolution_text = canvas.create_text(
    IMAGE_X2 - 24,
    IMAGE_Y1 + 22,
    text="SOURCE: STANDBY",
    anchor="e",
    fill=GREY_DARK,
    font=TINY_FONT
)

placeholder_halo = canvas.create_oval(
    (IMAGE_X1 + IMAGE_X2) // 2 - 52,
    (IMAGE_Y1 + IMAGE_Y2) // 2 - 73,
    (IMAGE_X1 + IMAGE_X2) // 2 + 52,
    (IMAGE_Y1 + IMAGE_Y2) // 2 + 31,
    fill="",
    outline="#153A50",
    width=2
)

placeholder_icon = canvas.create_text(
    (IMAGE_X1 + IMAGE_X2) // 2,
    (IMAGE_Y1 + IMAGE_Y2) // 2 - 21,
    text="+",
    fill="#2C5870",
    font=("Segoe UI", 58)
)

placeholder_text = canvas.create_text(
    (IMAGE_X1 + IMAGE_X2) // 2,
    (IMAGE_Y1 + IMAGE_Y2) // 2 + 63,
    text="UPLOAD TRAFFIC SIGN IMAGE",
    fill=LIGHT,
    font=("Segoe UI", 10, "bold")
)

placeholder_sub = canvas.create_text(
    (IMAGE_X1 + IMAGE_X2) // 2,
    (IMAGE_Y1 + IMAGE_Y2) // 2 + 88,
    text="JPG  •  JPEG  •  PNG  •  BMP  •  GIF",
    fill=GREY_DARK,
    font=TINY_FONT
)

scan_line = canvas.create_line(
    IMAGE_X1 + 18,
    IMAGE_Y1 + 48,
    IMAGE_X2 - 18,
    IMAGE_Y1 + 48,
    fill=CYAN,
    width=2,
    state="hidden"
)

scan_glow = canvas.create_line(
    IMAGE_X1 + 18,
    IMAGE_Y1 + 52,
    IMAGE_X2 - 18,
    IMAGE_Y1 + 52,
    fill="#0A5B60",
    width=1,
    state="hidden"
)

image_label = tk.Label(
    top,
    bg=PANEL_INNER,
    bd=0,
    highlightthickness=0
)

image_window = canvas.create_window(
    (IMAGE_X1 + IMAGE_X2) // 2,
    (IMAGE_Y1 + IMAGE_Y2) // 2 + 10,
    window=image_label
)


# ============================================================
# MODEL STATUS PANEL
# ============================================================

rounded_box(
    STATUS_X1 + 5,
    STATUS_Y1 + 7,
    STATUS_X2 + 5,
    STATUS_Y2 + 7,
    13,
    SHADOW
)

rounded_box(
    STATUS_X1,
    STATUS_Y1,
    STATUS_X2,
    STATUS_Y2,
    13,
    PANEL_TOP,
    BORDER,
    1
)

status_halo = canvas.create_oval(
    STATUS_X1 + 17,
    STATUS_Y1 + 24,
    STATUS_X1 + 49,
    STATUS_Y1 + 56,
    fill="",
    outline="#0B684E",
    width=1
)

status_dot = canvas.create_oval(
    STATUS_X1 + 27,
    STATUS_Y1 + 34,
    STATUS_X1 + 39,
    STATUS_Y1 + 46,
    fill=GREEN,
    outline=""
)

status_text = canvas.create_text(
    STATUS_X1 + 61,
    STATUS_Y1 + 31,
    text="MODEL READY",
    anchor="w",
    fill=GREEN,
    font=("Segoe UI", 9, "bold")
)

status_subtext = canvas.create_text(
    STATUS_X1 + 61,
    STATUS_Y1 + 52,
    text="CNN recognition engine active",
    anchor="w",
    fill=GREY,
    font=SMALL_FONT
)

canvas.create_text(
    STATUS_X2 - 18,
    STATUS_Y1 + 31,
    text="43",
    anchor="e",
    fill=WHITE,
    font=("Segoe UI", 16, "bold")
)

canvas.create_text(
    STATUS_X2 - 18,
    STATUS_Y1 + 53,
    text="SIGN CLASSES",
    anchor="e",
    fill=GREY_DARK,
    font=TINY_FONT
)


# ============================================================
# RESULT PANEL
# ============================================================

rounded_box(
    RESULT_X1 + 7,
    RESULT_Y1 + 10,
    RESULT_X2 + 7,
    RESULT_Y2 + 10,
    17,
    SHADOW
)

rounded_box(
    RESULT_X1,
    RESULT_Y1,
    RESULT_X2,
    RESULT_Y2,
    17,
    PANEL_INNER,
    CYAN,
    1
)

canvas.create_line(
    RESULT_X1 + 30,
    RESULT_Y1 + 2,
    RESULT_X2 - 30,
    RESULT_Y1 + 2,
    fill=CYAN_BRIGHT,
    width=1
)

canvas.create_text(
    (RESULT_X1 + RESULT_X2) // 2,
    RESULT_Y1 + 30,
    text="DETECTED SIGN",
    fill=GREY,
    font=("Segoe UI", 8, "bold")
)

canvas.create_text(
    RESULT_X1 + 22,
    RESULT_Y1 + 27,
    text="01",
    anchor="w",
    fill=GREY_DARK,
    font=DIGITAL_FONT
)

result_label = tk.Label(
    top,
    text="Waiting for image...",
    bg=PANEL_INNER,
    fg=GREY_DARK,
    font=RESULT_FONT,
    wraplength=360,
    justify="center",
    bd=0
)

result_window = canvas.create_window(
    (RESULT_X1 + RESULT_X2) // 2,
    RESULT_Y1 + 91,
    window=result_label
)

canvas.create_text(
    RESULT_X1 + 26,
    RESULT_Y1 + 150,
    text="CONFIDENCE LEVEL",
    anchor="w",
    fill=GREY,
    font=TINY_FONT
)

confidence_text = canvas.create_text(
    RESULT_X2 - 26,
    RESULT_Y1 + 150,
    text="--",
    anchor="e",
    fill=LIGHT,
    font=DIGITAL_FONT
)

CONFIDENCE_BAR_X1 = RESULT_X1 + 26
CONFIDENCE_BAR_Y1 = RESULT_Y1 + 169
CONFIDENCE_BAR_X2 = RESULT_X2 - 26
CONFIDENCE_BAR_Y2 = CONFIDENCE_BAR_Y1 + 10

rounded_box(
    CONFIDENCE_BAR_X1,
    CONFIDENCE_BAR_Y1,
    CONFIDENCE_BAR_X2,
    CONFIDENCE_BAR_Y2,
    5,
    "#07101E",
    BORDER,
    1
)

confidence_fill = canvas.create_rectangle(
    CONFIDENCE_BAR_X1 + 2,
    CONFIDENCE_BAR_Y1 + 2,
    CONFIDENCE_BAR_X1 + 2,
    CONFIDENCE_BAR_Y2 - 2,
    fill=CYAN,
    outline=""
)

canvas.create_text(
    RESULT_X1 + 26,
    RESULT_Y2 - 19,
    text="ENGINE // TRAFFIC_CLASSIFIER.H5",
    anchor="w",
    fill=GREY_DARK,
    font=TINY_FONT
)


# ============================================================
# 3-D BUTTON
# ============================================================

class ModernButton(tk.Canvas):

    def __init__(
        self,
        parent,
        text,
        command,
        width=270,
        height=62,
        accent=CYAN
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=BG_BOTTOM,
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )

        self.text = text
        self.command = command
        self.button_width = width
        self.button_height = height
        self.accent = accent

        self.hover = False
        self.pressed = False
        self.enabled = True

        self.draw()

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

        self.bind_all("<Return>", self.keyboard_release)

    def create_rounded_polygon(
        self,
        x1,
        y1,
        x2,
        y2,
        radius,
        fill,
        outline="",
        width=1
    ):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]

        return self.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=outline,
            width=width
        )

    def draw(self):
        self.delete("all")

        offset = 3 if self.pressed else 0

        if not self.enabled:
            face = "#26364A"
            top_color = "#34475E"
            text_color = GREY
            border_color = GREY_DARK
        elif self.hover:
            face = blend_colors(self.accent, WHITE, 0.15)
            top_color = CYAN_BRIGHT
            text_color = WHITE
            border_color = CYAN_BRIGHT
        else:
            face = blend_colors(self.accent, BG_TOP, 0.25)
            top_color = blend_colors(self.accent, WHITE, 0.25)
            text_color = WHITE
            border_color = self.accent

        # Deep shadow.
        self.create_rounded_polygon(
            6,
            10,
            self.button_width - 4,
            self.button_height - 2,
            12,
            SHADOW
        )

        # Bottom bevel.
        self.create_rounded_polygon(
            4,
            6 + offset,
            self.button_width - 6,
            self.button_height - 7 + offset,
            12,
            blend_colors(face, BG_TOP, 0.42),
            border_color,
            1
        )

        # Main button face.
        self.create_rounded_polygon(
            7,
            3 + offset,
            self.button_width - 9,
            self.button_height - 12 + offset,
            10,
            face,
            border_color,
            1
        )

        # Top illumination.
        self.create_line(
            21,
            7 + offset,
            self.button_width - 23,
            7 + offset,
            fill=top_color,
            width=2
        )

        # Left icon plate.
        self.create_oval(
            24,
            18 + offset,
            44,
            38 + offset,
            fill=PANEL_DARK,
            outline=top_color,
            width=1
        )

        self.create_oval(
            31,
            25 + offset,
            37,
            31 + offset,
            fill=top_color,
            outline=""
        )

        self.create_text(
            self.button_width // 2 + 9,
            28 + offset,
            text=self.text,
            fill=text_color,
            font=BUTTON_FONT
        )

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.configure(
            cursor="hand2" if enabled else "arrow"
        )
        self.draw()

    def on_enter(self, event):
        if not self.enabled:
            return

        self.hover = True
        self.draw()

    def on_leave(self, event):
        self.hover = False
        self.pressed = False
        self.draw()

    def on_press(self, event):
        if not self.enabled:
            return

        self.pressed = True
        self.draw()

    def on_release(self, event):
        if not self.enabled:
            return

        was_pressed = self.pressed
        self.pressed = False
        self.draw()

        if was_pressed:
            self.command()

    def keyboard_release(self, event):
        return


# ============================================================
# UI STATE
# ============================================================

selected_file = None
scan_active = False
scan_position = IMAGE_Y1 + 48
scan_direction = 1
pulse_value = 0
pulse_direction = 1
resize_job = None


def reset_confidence_bar():
    canvas.coords(
        confidence_fill,
        CONFIDENCE_BAR_X1 + 2,
        CONFIDENCE_BAR_Y1 + 2,
        CONFIDENCE_BAR_X1 + 2,
        CONFIDENCE_BAR_Y2 - 2
    )

    canvas.itemconfigure(
        confidence_text,
        text="--",
        fill=LIGHT
    )


def update_confidence_bar(confidence):
    safe_confidence = max(0.0, min(100.0, confidence))

    maximum_width = (
        CONFIDENCE_BAR_X2
        - CONFIDENCE_BAR_X1
        - 4
    )

    fill_width = maximum_width * (
        safe_confidence / 100.0
    )

    canvas.coords(
        confidence_fill,
        CONFIDENCE_BAR_X1 + 2,
        CONFIDENCE_BAR_Y1 + 2,
        CONFIDENCE_BAR_X1 + 2 + fill_width,
        CONFIDENCE_BAR_Y2 - 2
    )

    if safe_confidence >= 80:
        bar_color = GREEN
    elif safe_confidence >= 50:
        bar_color = CYAN
    else:
        bar_color = YELLOW

    canvas.itemconfigure(
        confidence_fill,
        fill=bar_color
    )

    canvas.itemconfigure(
        confidence_text,
        text=f"{safe_confidence:.2f}%",
        fill=bar_color
    )


def set_status(title, subtitle, color):
    canvas.itemconfigure(
        status_text,
        text=title,
        fill=color
    )

    canvas.itemconfigure(
        status_subtext,
        text=subtitle
    )

    canvas.itemconfigure(
        status_dot,
        fill=color
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():
    global selected_file
    global scan_active
    global scan_position

    file_path = filedialog.askopenfilename(
        title="Select Traffic Sign Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.gif"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    if not file_path:
        return

    try:
        selected_file = file_path

        uploaded = Image.open(file_path)
        uploaded = uploaded.convert("RGB")

        original_width, original_height = uploaded.size

        uploaded.thumbnail(
            (
                IMAGE_X2 - IMAGE_X1 - 58,
                IMAGE_Y2 - IMAGE_Y1 - 68
            ),
            Image.Resampling.LANCZOS
        )

        preview = ImageTk.PhotoImage(uploaded)

        image_label.configure(
            image=preview,
            bg=PANEL_INNER
        )

        image_label.image = preview

        # Hide the placeholder.
        canvas.itemconfigure(
            placeholder_halo,
            state="hidden"
        )

        canvas.itemconfigure(
            placeholder_icon,
            state="hidden"
        )

        canvas.itemconfigure(
            placeholder_text,
            state="hidden"
        )

        canvas.itemconfigure(
            placeholder_sub,
            state="hidden"
        )

        canvas.itemconfigure(
            resolution_text,
            text=f"SOURCE: {original_width} x {original_height}",
            fill=CYAN
        )

        set_status(
            "IMAGE LOADED",
            "Input frame ready for analysis",
            CYAN
        )

        result_label.configure(
            text="Ready to classify",
            fg=CYAN
        )

        reset_confidence_bar()

        scan_position = IMAGE_Y1 + 48
        scan_active = True

        canvas.itemconfigure(
            scan_line,
            state="normal"
        )

        canvas.itemconfigure(
            scan_glow,
            state="normal"
        )

        print(
            "Image selected:",
            file_path
        )

    except Exception as e:
        print(
            "Image loading error:",
            e
        )

        result_label.configure(
            text="Unable to open image",
            fg=RED
        )

        set_status(
            "IMAGE ERROR",
            "Selected file could not be opened",
            RED
        )


# ============================================================
# CLASSIFY IMAGE
# ============================================================

def classify():
    global selected_file
    global scan_active

    if selected_file is None:
        result_label.configure(
            text="Please upload an image first",
            fg=RED
        )

        set_status(
            "INPUT REQUIRED",
            "Upload a traffic sign image first",
            RED
        )

        return

    try:
        # Update interface.
        result_label.configure(
            text="ANALYZING...",
            fg=CYAN
        )

        reset_confidence_bar()

        set_status(
            "ANALYZING IMAGE...",
            "Neural inference in progress",
            CYAN
        )

        classify_button.set_enabled(False)
        upload_button.set_enabled(False)

        scan_active = True

        canvas.itemconfigure(
            scan_line,
            state="normal"
        )

        canvas.itemconfigure(
            scan_glow,
            state="normal"
        )

        top.update_idletasks()


        # ====================================================
        # ORIGINAL MODEL LOGIC — UNCHANGED
        # ====================================================

        image = Image.open(
            selected_file
        )

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (30, 30)
        )

        image = np.array(
            image,
            dtype=np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        print(
            "Input shape:",
            image.shape
        )

        pred_probs = model.predict(
            image,
            verbose=0
        )[0]

        pred = np.argmax(
            pred_probs
        )

        class_number = pred + 1

        sign = classes.get(
            class_number,
            "Unknown traffic sign"
        )

        confidence = float(
            pred_probs[pred] * 100
        )

        print(
            "Predicted class:",
            class_number
        )

        print(
            "Traffic sign:",
            sign
        )

        print(
            "Confidence:",
            confidence
        )


        # ====================================================
        # UPDATE DISPLAY
        # ====================================================

        result_label.configure(
            text=sign,
            fg=CYAN_BRIGHT
        )

        set_status(
            "PREDICTION COMPLETE",
            f"Class ID {class_number:02d} successfully identified",
            GREEN
        )

        update_confidence_bar(confidence)

        scan_active = False

        canvas.itemconfigure(
            scan_line,
            state="hidden"
        )

        canvas.itemconfigure(
            scan_glow,
            state="hidden"
        )

    except Exception as e:
        print(
            "Prediction error:",
            e
        )

        result_label.configure(
            text="Unable to classify image",
            fg=RED
        )

        set_status(
            "PREDICTION ERROR",
            "Recognition engine could not complete",
            RED
        )

        scan_active = False

        canvas.itemconfigure(
            scan_line,
            state="hidden"
        )

        canvas.itemconfigure(
            scan_glow,
            state="hidden"
        )

    finally:
        classify_button.set_enabled(True)
        upload_button.set_enabled(True)


# ============================================================
# BUTTONS
# ============================================================

upload_button = ModernButton(
    top,
    "UPLOAD IMAGE",
    upload_image,
    width=280,
    height=64,
    accent=CYAN
)

upload_button_window = canvas.create_window(
    515,
    BUTTON_Y,
    window=upload_button
)

classify_button = ModernButton(
    top,
    "CLASSIFY IMAGE",
    classify,
    width=280,
    height=64,
    accent=BLUE
)

classify_button_window = canvas.create_window(
    925,
    BUTTON_Y,
    window=classify_button
)


# ============================================================
# FOOTER
# ============================================================

canvas.create_line(
    60,
    836,
    1380,
    836,
    fill=BORDER,
    width=1
)

canvas.create_text(
    60,
    860,
    text="TRAFFIC SIGN RECOGNITION  //  CONVOLUTIONAL NEURAL NETWORK",
    anchor="w",
    fill=GREY_DARK,
    font=("Segoe UI", 8, "bold")
)

canvas.create_text(
    1380,
    860,
    text="VISION ENGINE  •  READY",
    anchor="e",
    fill=GREEN,
    font=("Segoe UI", 8, "bold")
)


# ============================================================
# REAL-TIME ANIMATIONS
# ============================================================

def update_clock():
    current_time = datetime.now().strftime("%H:%M:%S")

    canvas.itemconfigure(
        clock_text,
        text=current_time
    )

    top.after(
        1000,
        update_clock
    )


def animate_status_pulse():
    global pulse_value
    global pulse_direction

    pulse_value += pulse_direction

    if pulse_value >= 8:
        pulse_direction = -1

    elif pulse_value <= 0:
        pulse_direction = 1

    ratio = pulse_value / 8.0

    halo_color = blend_colors(
        "#0A3B35",
        GREEN,
        ratio * 0.6
    )

    canvas.itemconfigure(
        system_halo,
        outline=halo_color
    )

    canvas.itemconfigure(
        status_halo,
        outline=halo_color
    )

    top.after(
        90,
        animate_status_pulse
    )


def animate_header():
    phase = (
        datetime.now().microsecond
        / 1000000.0
    )

    glow_ratio = abs(
        0.5 - phase
    ) * 2

    glow_color = blend_colors(
        "#063A42",
        "#0C7A82",
        glow_ratio
    )

    canvas.itemconfigure(
        header_title_glow,
        fill=glow_color
    )

    top.after(
        80,
        animate_header
    )


def animate_scanner():
    global scan_position
    global scan_direction

    if scan_active:
        scan_position += 4 * scan_direction

        minimum_y = IMAGE_Y1 + 48
        maximum_y = IMAGE_Y2 - 24

        if scan_position >= maximum_y:
            scan_position = maximum_y
            scan_direction = -1

        elif scan_position <= minimum_y:
            scan_position = minimum_y
            scan_direction = 1

        canvas.coords(
            scan_line,
            IMAGE_X1 + 18,
            scan_position,
            IMAGE_X2 - 18,
            scan_position
        )

        canvas.coords(
            scan_glow,
            IMAGE_X1 + 18,
            scan_position + 5,
            IMAGE_X2 - 18,
            scan_position + 5
        )

        canvas.tag_raise(scan_line)
        canvas.tag_raise(scan_glow)

    top.after(
        24,
        animate_scanner
    )


# ============================================================
# RESPONSIVE CENTERING
# ============================================================

def redraw_after_resize():
    width = max(
        canvas.winfo_width(),
        DESIGN_WIDTH
    )

    height = max(
        canvas.winfo_height(),
        DESIGN_HEIGHT
    )

    draw_background(width, height)

    x_offset = max(
        (canvas.winfo_width() - DESIGN_WIDTH) / 2,
        0
    )

    canvas.configure(
        scrollregion=(
            -x_offset,
            0,
            DESIGN_WIDTH + x_offset,
            DESIGN_HEIGHT
        )
    )


def keep_layout(event):
    global resize_job

    if event.widget != canvas:
        return

    if resize_job is not None:
        top.after_cancel(resize_job)

    resize_job = top.after(
        120,
        redraw_after_resize
    )


canvas.bind(
    "<Configure>",
    keep_layout
)


# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================

top.bind(
    "<Control-o>",
    lambda event: upload_image()
)

top.bind(
    "<Control-O>",
    lambda event: upload_image()
)

top.bind(
    "<Return>",
    lambda event: classify()
)


# ============================================================
# START ANIMATIONS
# ============================================================

update_clock()
animate_status_pulse()
animate_header()
animate_scanner()


# ============================================================
# START APPLICATION
# ============================================================

top.mainloop()
