import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont

from interface import MainWindow


def create_splash_screen():
    """Create a splash screen with loading animation"""
    # Create a simple splash widget
    splash_widget = QWidget()
    splash_widget.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash_widget.setAttribute(Qt.WA_TranslucentBackground)

    # Create layout with margins
    layout = QVBoxLayout(splash_widget)
    layout.setAlignment(Qt.AlignCenter)
    layout.setContentsMargins(30, 30, 30, 30)
    layout.setSpacing(30)

    # Title label
    title = QLabel("SVAS")
    title.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 48px;
            font-weight: bold;
            background-color: transparent;
            padding: 10px;
        }
    """)
    title.setAlignment(Qt.AlignCenter)

    # Subtitle
    subtitle = QLabel("Students Video Analytics Suite")
    subtitle.setStyleSheet("""
        QLabel {
            color: rgba(255, 255, 255, 180);
            font-size: 18px;
            background-color: transparent;
            padding: 5px;
        }
    """)
    subtitle.setAlignment(Qt.AlignCenter)

    # Loading label with spinner
    loading = QLabel()
    loading.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 24px;
            background-color: transparent;
            padding: 20px;
        }
    """)
    loading.setAlignment(Qt.AlignCenter)
    loading.setMinimumHeight(80)

    # Animated spinner
    spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    rotation = [0]  # Use list to allow modification in nested function

    def update_spinner():
        rotation[0] = (rotation[0] + 1) % len(spinners)
        loading.setText(f"{spinners[rotation[0]]}  Loading...")

    timer = QTimer()
    timer.timeout.connect(update_spinner)
    timer.start(80)
    update_spinner()  # Show initial state

    # Store timer reference so it doesn't get garbage collected
    loading.timer = timer

    layout.addStretch()
    layout.addWidget(title)
    layout.addWidget(subtitle)
    layout.addSpacing(20)
    layout.addWidget(loading)
    layout.addStretch()

    splash_widget.setFixedSize(600, 400)
    splash_widget.setStyleSheet("""
        QWidget {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(40, 40, 60, 250),
                stop:1 rgba(30, 30, 50, 250));
            border-radius: 20px;
            border: 2px solid rgba(255, 255, 255, 30);
        }
    """)

    return splash_widget


def main():
    app = QApplication(sys.argv)

    # Create and show splash screen
    splash = create_splash_screen()
    splash.show()

    # Center splash on screen
    screen_geometry = QApplication.desktop().availableGeometry()
    splash_geometry = splash.frameGeometry()
    splash_geometry.moveCenter(screen_geometry.center())
    splash.move(splash_geometry.topLeft())

    # Process events to show splash immediately
    QApplication.processEvents()

    # Create main window in main thread (required by Qt)
    window = MainWindow()

    # Stop splash animation
    if hasattr(splash.layout().itemAt(2).widget(), 'timer'):
        splash.layout().itemAt(2).widget().timer.stop()

    # Close splash and show main window
    splash.close()
    window.show()

    # Store window reference in app to prevent garbage collection
    app.main_window = window

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
