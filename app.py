import sys
import os
import logging
import threading
import time
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, 
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QSplitter, QTabWidget, QFrame, QStatusBar, QComboBox, QToolTip,
    QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QMetaObject, Q_ARG
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap, QImage

# Import the client functions
from client import (
    load_data, get_xgboost_params, AsyncXGBoostClient, logger
)
import flwr as fl
import xgboost as xgb
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from client import get_ensemble_feature_extractor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
import io
import cv2

# Constants for styling
MAIN_COLOR = "#3498db"  # Blue
SECONDARY_COLOR = "#2c3e50"  # Dark blue/gray
SUCCESS_COLOR = "#2ecc71"  # Green
ERROR_COLOR = "#e74c3c"  # Red
WARNING_COLOR = "#f39c12"  # Orange
BACKGROUND_COLOR = "#f5f5f5"  # Light gray
TEXT_COLOR = "#34495e"  # Dark gray

# Create a custom logger for the GUI
class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set up text colors for different log levels
        self.log_colors = {
            logging.DEBUG: "#7f8c8d",       # Gray
            logging.INFO: TEXT_COLOR,       # Normal text color
            logging.WARNING: WARNING_COLOR, # Orange
            logging.ERROR: ERROR_COLOR,     # Red
            logging.CRITICAL: "#9b59b6"     # Purple
        }

    def emit(self, record):
        msg = self.format(record)
        color = self.log_colors.get(record.levelno, TEXT_COLOR)
        formatted_msg = f'<span style="color:{color};">{msg}</span>'
        self.widget.append(formatted_msg)
        # Auto-scroll to the bottom
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


# Signal class for thread communication
class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    metrics = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    # Add signal for prediction results
    prediction_result = pyqtSignal(object, object, float, object)


class StyledProgressBar(QProgressBar):
    """Custom styled progress bar"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {SUCCESS_COLOR};
                width: 10px;
                margin: 0px;
            }}
        """)


class FederatedLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Federated XGBoost Learning Client")
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowIcon(QIcon('icon.png'))  # Add an icon if available
        
        # Set application style
        self.set_application_style()
        
        # Create a central widget with main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Add header/title
        self.create_header()
        
        # Create a splitter for the main interface
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_layout.addWidget(self.main_splitter)
        
        # Top section with configuration and controls
        self.top_section = QWidget()
        self.top_layout = QVBoxLayout(self.top_section)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs for different sections
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid #bdc3c7;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: white;
                border-bottom-color: white;
            }}
        """)
        
        # Create configuration tab
        self.config_tab = QWidget()
        self.create_input_form()
        self.tabs.addTab(self.config_tab, "Configuration")
        
        # Create metrics tab
        self.metrics_tab = QWidget()
        self.create_metrics_area()
        self.tabs.addTab(self.metrics_tab, "Training Metrics")
        
        # Create advanced metrics tab
        self.advanced_metrics_tab = QWidget()
        self.create_advanced_metrics_area()
        self.tabs.addTab(self.advanced_metrics_tab, "Advanced Metrics")
        
        # Create prediction tab
        self.prediction_tab = QWidget()
        self.create_prediction_area()
        self.tabs.addTab(self.prediction_tab, "Prediction")
        
        self.top_layout.addWidget(self.tabs)
        
        # Create control buttons
        self.create_control_buttons()
        
        # Add the top section to the splitter
        self.main_splitter.addWidget(self.top_section)
        
        # Log section
        self.log_section = QWidget()
        self.create_log_area()
        
        # Add the log section to the splitter
        self.main_splitter.addWidget(self.log_section)
        
        # Set initial splitter sizes (40% for top, 60% for logs)
        self.main_splitter.setSizes([400, 600])
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Initialize variables
        self.client = None
        self.worker_thread = None
        self.signals = WorkerSignals()
        
        # Connect signals
        self.signals.log.connect(self.update_log)
        self.signals.metrics.connect(self.update_metrics)
        self.signals.status_message.connect(self.statusBar.showMessage)
        self.signals.finished.connect(self.on_training_finished)
        
        # Set up custom logger
        self.setup_logger()

    def set_application_style(self):
        """Set global application styling"""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: white;
                color: {TEXT_COLOR};
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
            QGroupBox {{
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 15px;
                font-weight: bold;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {SECONDARY_COLOR};
            }}
            QPushButton {{
                background-color: {MAIN_COLOR};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
            QPushButton:pressed {{
                background-color: #1f618d;
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
            QLineEdit, QSpinBox, QComboBox {{
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QTextEdit {{
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                font-family: 'Consolas', 'Courier New', monospace;
            }}
        """)
    
    def create_header(self):
        """Create a header with title and description"""
        header = QFrame()
        header.setStyleSheet(f"""
            background-color: {SECONDARY_COLOR};
            border-radius: 5px;
            margin-bottom: 5px;
        """)
        header_layout = QVBoxLayout(header)
        
        # Title label
        title_label = QLabel("Federated XGBoost Learning Client")
        title_label.setStyleSheet(f"""
            color: white;
            font-size: 18px;
            font-weight: bold;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Description label
        desc_label = QLabel("Train machine learning models collaboratively across distributed clients")
        desc_label.setStyleSheet(f"""
            color: #ecf0f1;
            font-size: 12px;
        """)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        
        self.main_layout.addWidget(header)

    def create_input_form(self):
        form_layout = QVBoxLayout(self.config_tab)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(15)
        
        # Data configuration group
        data_group = QGroupBox("Data Configuration")
        data_layout = QFormLayout()
        data_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        data_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Data directory input
        self.data_dir_layout = QHBoxLayout()
        self.data_dir_input = QLineEdit()
        self.data_dir_input.setPlaceholderText("Select data directory...")
        self.data_dir_button = QPushButton("Browse")
        self.data_dir_button.setFixedWidth(80)
        self.data_dir_button.clicked.connect(self.browse_data_dir)
        self.data_dir_button.setIcon(QIcon("folder.png"))  # Add an icon if available
        self.data_dir_layout.addWidget(self.data_dir_input)
        self.data_dir_layout.addWidget(self.data_dir_button)
        data_layout.addRow(self.create_label_with_tooltip("Data Directory:", 
                                                          "Directory containing your training data"), 
                           self.data_dir_layout)
        
        # Feature selection
        self.feature_selection_checkbox = QCheckBox()
        data_layout.addRow(self.create_label_with_tooltip("Enable Feature Selection:", 
                                                          "Optimize model by selecting the most important features"), 
                           self.feature_selection_checkbox)
        
        data_group.setLayout(data_layout)
        form_layout.addWidget(data_group)
        
        # Server configuration group
        server_group = QGroupBox("Server Configuration")
        server_layout = QFormLayout()
        server_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        server_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Server address input
        self.server_address_input = QLineEdit("127.0.0.1:8080")
        server_layout.addRow(self.create_label_with_tooltip("Server Address:", 
                                                           "IP address and port of the federated learning server"), 
                             self.server_address_input)
        
        server_group.setLayout(server_layout)
        form_layout.addWidget(server_group)
        
        # Training configuration group
        training_group = QGroupBox("Training Configuration")
        training_layout = QFormLayout()
        training_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        training_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Number of local rounds
        self.local_rounds_input = QSpinBox()
        self.local_rounds_input.setMinimum(1)
        self.local_rounds_input.setMaximum(100)
        self.local_rounds_input.setValue(1)
        self.local_rounds_input.setStyleSheet("width: 60px;")
        training_layout.addRow(self.create_label_with_tooltip("Local Rounds:", 
                                                             "Number of training rounds to perform locally"), 
                               self.local_rounds_input)
        
        # Async configuration
        async_container = QWidget()
        self.async_layout = QHBoxLayout(async_container)
        self.async_layout.setContentsMargins(0, 0, 0, 0)
        
        self.async_checkbox = QCheckBox("Enable Async Mode")
        self.async_checkbox.setToolTip("Perform asynchronous updates to the server")
        
        self.async_interval_label = QLabel("Interval (seconds):")
        self.async_interval_input = QSpinBox()
        self.async_interval_input.setMinimum(1)
        self.async_interval_input.setMaximum(3600)
        self.async_interval_input.setValue(30)
        self.async_interval_input.setEnabled(False)
        self.async_checkbox.toggled.connect(self.async_interval_input.setEnabled)
        self.async_checkbox.toggled.connect(self.async_interval_label.setEnabled)
        
        self.async_layout.addWidget(self.async_checkbox)
        self.async_layout.addWidget(self.async_interval_label)
        self.async_layout.addWidget(self.async_interval_input)
        self.async_layout.addStretch()
        
        training_layout.addRow("", async_container)
        
        training_group.setLayout(training_layout)
        form_layout.addWidget(training_group)
        
        # Add stretch to push everything to the top
        form_layout.addStretch()

    def create_label_with_tooltip(self, text, tooltip):
        """Create a label with tooltip and info icon"""
        label = QLabel(text)
        label.setToolTip(tooltip)
        return label

    def create_log_area(self):
        layout = QVBoxLayout(self.log_section)
        layout.setContentsMargins(0, 0, 0, 0)
        
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        
        # Add log filter controls
        log_controls = QHBoxLayout()
        
        log_level_label = QLabel("Log Level:")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self.set_log_level)
        
        clear_log_button = QPushButton("Clear Logs")
        clear_log_button.setFixedWidth(100)
        clear_log_button.clicked.connect(self.clear_logs)
        
        log_controls.addWidget(log_level_label)
        log_controls.addWidget(self.log_level_combo)
        log_controls.addStretch()
        log_controls.addWidget(clear_log_button)
        
        log_layout.addLayout(log_controls)
        
        # Log text area with HTML support
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_text.setStyleSheet("""
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
        """)
        self.log_text.document().setMaximumBlockCount(5000)  # Limit number of lines
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def create_metrics_area(self):
        metrics_layout = QVBoxLayout(self.metrics_tab)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(15)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_layout = QHBoxLayout()
        self.progress_bar = StyledProgressBar()
        self.progress_label = QLabel("0%")
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.progress_label)
        progress_layout.addLayout(self.progress_layout)
        
        # Status label
        self.training_status_label = QLabel("Status: Idle")
        self.training_status_label.setStyleSheet(f"font-weight: bold; color: {TEXT_COLOR};")
        progress_layout.addWidget(self.training_status_label)
        
        progress_group.setLayout(progress_layout)
        metrics_layout.addWidget(progress_group)
        
        # Model performance metrics
        metrics_group = QGroupBox("Model Performance")
        metrics_form_layout = QFormLayout()
        metrics_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        metrics_form_layout.setVerticalSpacing(10)
        
        # Create metric displays with consistent styling
        self.accuracy_label = self.create_metric_label("N/A")
        self.auc_label = self.create_metric_label("N/A")
        self.training_time_label = self.create_metric_label("N/A")
        self.model_size_label = self.create_metric_label("N/A")
        
        metrics_form_layout.addRow(QLabel("Accuracy:"), self.accuracy_label)
        metrics_form_layout.addRow(QLabel("AUC:"), self.auc_label)
        metrics_form_layout.addRow(QLabel("Training Time:"), self.training_time_label)
        metrics_form_layout.addRow(QLabel("Model Size:"), self.model_size_label)
        
        metrics_group.setLayout(metrics_form_layout)
        metrics_layout.addWidget(metrics_group)
        
        # Add stretch to push groups to the top
        metrics_layout.addStretch()

    def create_advanced_metrics_area(self):
        """Create the advanced metrics area with federated learning specific metrics"""
        metrics_layout = QVBoxLayout(self.advanced_metrics_tab)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(15)
        
        # Classification metrics
        classification_group = QGroupBox("Classification Metrics")
        classification_layout = QFormLayout()
        classification_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        classification_layout.setVerticalSpacing(10)
        
        # Create metric displays
        self.precision_label = self.create_metric_label("N/A")
        self.recall_label = self.create_metric_label("N/A")
        self.f1_score_label = self.create_metric_label("N/A")
        
        classification_layout.addRow(self.create_label_with_tooltip("Precision:", 
                                    "Ratio of true positives to predicted positives"), 
                                    self.precision_label)
        classification_layout.addRow(self.create_label_with_tooltip("Recall:", 
                                    "Ratio of true positives to actual positives"),
                                    self.recall_label)
        classification_layout.addRow(self.create_label_with_tooltip("F1 Score:", 
                                    "Harmonic mean of precision and recall"),
                                    self.f1_score_label)
        
        classification_group.setLayout(classification_layout)
        metrics_layout.addWidget(classification_group)
        
        # Confusion matrix
        confusion_group = QGroupBox("Confusion Matrix")
        confusion_layout = QGridLayout()
        
        # Header labels
        confusion_layout.addWidget(QLabel(""), 0, 0)
        confusion_layout.addWidget(self.create_centered_label("Predicted\nPositive"), 0, 1)
        confusion_layout.addWidget(self.create_centered_label("Predicted\nNegative"), 0, 2)
        confusion_layout.addWidget(self.create_centered_label("Actual\nPositive"), 1, 0)
        confusion_layout.addWidget(self.create_centered_label("Actual\nNegative"), 2, 0)
        
        # Confusion matrix cells
        self.tp_label = self.create_metric_label("N/A")
        self.fp_label = self.create_metric_label("N/A")
        self.fn_label = self.create_metric_label("N/A")
        self.tn_label = self.create_metric_label("N/A")
        
        confusion_layout.addWidget(self.tp_label, 1, 1)  # True Positive
        confusion_layout.addWidget(self.fn_label, 1, 2)  # False Negative
        confusion_layout.addWidget(self.fp_label, 2, 1)  # False Positive
        confusion_layout.addWidget(self.tn_label, 2, 2)  # True Negative
        
        confusion_group.setLayout(confusion_layout)
        metrics_layout.addWidget(confusion_group)
        
        # Federated learning specific metrics
        federated_group = QGroupBox("Federated Learning Metrics")
        federated_layout = QFormLayout()
        federated_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        federated_layout.setVerticalSpacing(10)
        
        # Create federated metrics
        self.rounds_label = self.create_metric_label("N/A")
        self.comm_overhead_label = self.create_metric_label("N/A")
        self.convergence_rate_label = self.create_metric_label("N/A")
        
        federated_layout.addRow(self.create_label_with_tooltip("Global Rounds:", 
                               "Number of global aggregation rounds completed"), 
                               self.rounds_label)
        federated_layout.addRow(self.create_label_with_tooltip("Communication Overhead:", 
                               "Data transferred between client and server (KB)"),
                               self.comm_overhead_label)
        federated_layout.addRow(self.create_label_with_tooltip("Convergence Rate:", 
                               "How quickly the model reaches optimal performance"),
                               self.convergence_rate_label)
        
        federated_group.setLayout(federated_layout)
        metrics_layout.addWidget(federated_group)
        
        # Add stretch to push groups to the top
        metrics_layout.addStretch()

    def create_centered_label(self, text):
        """Create a centered label for the confusion matrix headers"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"""
            font-weight: bold;
            color: {SECONDARY_COLOR};
            padding: 5px;
        """)
        return label

    def create_metric_label(self, initial_value):
        """Create a styled label for metrics display"""
        label = QLabel(initial_value)
        label.setStyleSheet(f"""
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 14px;
            font-weight: bold;
            color: {SECONDARY_COLOR};
            padding: 5px;
            background-color: {BACKGROUND_COLOR};
            border-radius: 4px;
        """)
        return label

    def create_control_buttons(self):
        control_frame = QFrame()
        control_frame.setStyleSheet(f"""
            background-color: {BACKGROUND_COLOR};
            border-radius: 5px;
            padding: 10px;
        """)
        buttons_layout = QHBoxLayout(control_frame)
        buttons_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status indicator
        self.status_indicator = QLabel("Ready")
        self.status_indicator.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-weight: bold;
            padding: 8px;
        """)
        
        # Buttons
        self.start_button = QPushButton("Start Training")
        self.start_button.setIcon(QIcon("play.png"))  # Add an icon if available
        self.start_button.setStyleSheet(f"""
            background-color: {SUCCESS_COLOR};
            min-width: 120px;
        """)
        self.start_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(QIcon("stop.png"))  # Add an icon if available
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(f"""
            background-color: {ERROR_COLOR};
            min-width: 120px;
        """)
        self.stop_button.clicked.connect(self.stop_training)
        
        buttons_layout.addWidget(self.status_indicator)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        
        self.top_layout.addWidget(control_frame)

    def setup_logger(self):
        # Set up logging to text widget
        self.text_handler = QTextEditLogger(self.log_text)
        self.text_handler.setFormatter(logging.Formatter(
            '<b>%(asctime)s</b> - <span style="color:#8e44ad;">%(name)s</span> - %(levelname)s - %(message)s'
        ))
        
        # Add handler to logger
        logger.addHandler(self.text_handler)
        logger.setLevel(logging.INFO)

    def set_log_level(self, level_text):
        """Set the log level based on combo box selection"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level_text in level_map:
            logger.setLevel(level_map[level_text])
            self.update_log(f"<span style='color:{MAIN_COLOR};'>Log level set to {level_text}</span>")

    def clear_logs(self):
        """Clear the log text area"""
        self.log_text.clear()
        self.update_log("<span style='color:#3498db;'>Logs cleared</span>")

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Data Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if dir_path:
            self.data_dir_input.setText(dir_path)
            self.statusBar.showMessage(f"Data directory set to: {dir_path}", 3000)

    def update_log(self, message):
        self.log_text.append(message)

    def update_metrics(self, metrics):
        # Update accuracy with color-coded styling
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            accuracy_text = f"{accuracy:.4f}"
            color = self.get_metric_color(accuracy, 0.7, 0.9)
            self.accuracy_label.setText(accuracy_text)
            self.accuracy_label.setStyleSheet(f"""
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 14px;
                font-weight: bold;
                color: {color};
                padding: 5px;
                background-color: {BACKGROUND_COLOR};
                border-radius: 4px;
            """)
        
        # Look for AUC metric (it might have different names)
        for key in metrics:
            if 'auc' in key.lower():
                auc = metrics[key]
                auc_text = f"{auc:.4f}"
                color = self.get_metric_color(auc, 0.7, 0.9)
                self.auc_label.setText(auc_text)
                self.auc_label.setStyleSheet(f"""
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 14px;
                    font-weight: bold;
                    color: {color};
                    padding: 5px;
                    background-color: {BACKGROUND_COLOR};
                    border-radius: 4px;
                """)
                break
        
        if 'training_time' in metrics:
            self.training_time_label.setText(f"{metrics['training_time']:.2f} seconds")
        
        if 'model_size' in metrics:
            size_kb = metrics['model_size'] / 1024
            self.model_size_label.setText(f"{size_kb:.2f} KB")
            
            # Update communication overhead as a factor of model size
            if hasattr(self, 'comm_overhead_label'):
                # Estimate communication overhead (model size * rounds)
                comm_overhead = size_kb
                if hasattr(self.client, 'current_round'):
                    comm_overhead *= self.client.current_round
                self.comm_overhead_label.setText(f"{comm_overhead:.2f} KB")
            
        # Update advanced metrics if available
        if hasattr(self, 'precision_label'):
            if 'precision' in metrics:
                precision = metrics['precision']
                precision_text = f"{precision:.4f}"
                color = self.get_metric_color(precision, 0.7, 0.9)
                self.precision_label.setText(precision_text)
                self.precision_label.setStyleSheet(f"""
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 14px;
                    font-weight: bold;
                    color: {color};
                    padding: 5px;
                    background-color: {BACKGROUND_COLOR};
                    border-radius: 4px;
                """)
            
            if 'recall' in metrics:
                recall = metrics['recall']
                recall_text = f"{recall:.4f}"
                color = self.get_metric_color(recall, 0.7, 0.9)
                self.recall_label.setText(recall_text)
                self.recall_label.setStyleSheet(f"""
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 14px;
                    font-weight: bold;
                    color: {color};
                    padding: 5px;
                    background-color: {BACKGROUND_COLOR};
                    border-radius: 4px;
                """)
                
            if 'f1_score' in metrics:
                f1_score = metrics['f1_score']
                f1_score_text = f"{f1_score:.4f}"
                color = self.get_metric_color(f1_score, 0.7, 0.9)
                self.f1_score_label.setText(f1_score_text)
                self.f1_score_label.setStyleSheet(f"""
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 14px;
                    font-weight: bold;
                    color: {color};
                    padding: 5px;
                    background-color: {BACKGROUND_COLOR};
                    border-radius: 4px;
                """)
                
        # Update confusion matrix
        if hasattr(self, 'tp_label'):
            if 'true_positives' in metrics:
                self.tp_label.setText(str(metrics['true_positives']))
            if 'false_positives' in metrics:
                self.fp_label.setText(str(metrics['false_positives']))
            if 'false_negatives' in metrics:
                self.fn_label.setText(str(metrics['false_negatives']))
            if 'true_negatives' in metrics:
                self.tn_label.setText(str(metrics['true_negatives']))
        
        # Update federated learning metrics
        if hasattr(self, 'rounds_label') and hasattr(self.client, 'current_round'):
            self.rounds_label.setText(str(self.client.current_round))
            
            # Calculate convergence rate if we have accuracy history
            if hasattr(self.client, 'accuracy_history') and len(self.client.accuracy_history) > 1:
                # Simple convergence rate: accuracy improvement per round
                accuracy_improvement = self.client.accuracy_history[-1] - self.client.accuracy_history[0]
                rounds_elapsed = len(self.client.accuracy_history) - 1
                convergence_rate = accuracy_improvement / rounds_elapsed if rounds_elapsed > 0 else 0
                self.convergence_rate_label.setText(f"{convergence_rate:.4f}/round")
            else:
                # Store accuracy history if it doesn't exist
                if 'accuracy' in metrics:
                    if not hasattr(self.client, 'accuracy_history'):
                        self.client.accuracy_history = []
                    self.client.accuracy_history.append(metrics['accuracy'])

    def get_metric_color(self, value, threshold_yellow, threshold_green):
        """Return a color based on metric value thresholds"""
        if value < threshold_yellow:
            return ERROR_COLOR  # Red for low values
        elif value < threshold_green:
            return WARNING_COLOR  # Yellow for medium values
        else:
            return SUCCESS_COLOR  # Green for high values

    def start_training(self):
        # Validate inputs
        data_dir = self.data_dir_input.text()
        if not os.path.isdir(data_dir):
            self.update_log(f"<span style='color:{ERROR_COLOR};'>Error: Invalid data directory</span>")
            self.statusBar.showMessage("Error: Invalid data directory", 5000)
            return
        
        server_address = self.server_address_input.text()
        if not server_address:
            self.update_log(f"<span style='color:{ERROR_COLOR};'>Error: Server address is required</span>")
            self.statusBar.showMessage("Error: Server address is required", 5000)
            return
        
        # Get other parameters
        num_local_rounds = self.local_rounds_input.value()
        async_interval = self.async_interval_input.value() if self.async_checkbox.isChecked() else None
        feature_selection = self.feature_selection_checkbox.isChecked()
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.training_status_label.setText("Status: Training")
        self.training_status_label.setStyleSheet(f"font-weight: bold; color: {MAIN_COLOR};")
        self.status_indicator.setText("Training")
        self.status_indicator.setStyleSheet(f"""
            color: {MAIN_COLOR};
            font-weight: bold;
            padding: 8px;
        """)
        
        # Reset metrics
        self.reset_metrics()
        
        # Start training in a separate thread
        self.worker_thread = threading.Thread(
            target=self.run_training,
            args=(data_dir, server_address, num_local_rounds, async_interval, feature_selection),
            daemon=True
        )
        self.worker_thread.start()
        
        # Start a timer to update the UI
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)  # Update every 500ms
        
        self.statusBar.showMessage("Training started", 3000)

    def reset_metrics(self):
        """Reset all metrics displays to initial state"""
        default_style = f"""
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 14px;
            font-weight: bold;
            color: {SECONDARY_COLOR};
            padding: 5px;
            background-color: {BACKGROUND_COLOR};
            border-radius: 4px;
        """
        
        # Reset basic metrics
        self.accuracy_label.setText("N/A")
        self.accuracy_label.setStyleSheet(default_style)
        
        self.auc_label.setText("N/A")
        self.auc_label.setStyleSheet(default_style)
        
        self.training_time_label.setText("N/A")
        self.model_size_label.setText("N/A")
        
        # Reset advanced metrics
        if hasattr(self, 'precision_label'):
            self.precision_label.setText("N/A")
            self.precision_label.setStyleSheet(default_style)
            
        if hasattr(self, 'recall_label'):
            self.recall_label.setText("N/A")
            self.recall_label.setStyleSheet(default_style)
            
        if hasattr(self, 'f1_score_label'):
            self.f1_score_label.setText("N/A")
            self.f1_score_label.setStyleSheet(default_style)
            
        # Reset confusion matrix
        if hasattr(self, 'tp_label'):
            self.tp_label.setText("N/A")
            self.fp_label.setText("N/A")
            self.fn_label.setText("N/A")
            self.tn_label.setText("N/A")
            
        # Reset federated metrics
        if hasattr(self, 'rounds_label'):
            self.rounds_label.setText("N/A")
            self.comm_overhead_label.setText("N/A")
            self.convergence_rate_label.setText("N/A")
        
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")

    def run_training(self, data_dir, server_address, num_local_rounds, async_interval, feature_selection):
        try:
            self.signals.status_message.emit("Initializing training...")
            
            # Log the training parameters with formatted output
            self.signals.log.emit("<span style='color:#2980b9; font-weight:bold;'>Starting training with parameters:</span>")
            self.signals.log.emit(f"- Data Directory: <b>{data_dir}</b>")
            self.signals.log.emit(f"- Server Address: <b>{server_address}</b>")
            self.signals.log.emit(f"- Local Rounds: <b>{num_local_rounds}</b>")
            self.signals.log.emit(f"- Async Interval: <b>{async_interval if async_interval else 'None'}</b>")
            self.signals.log.emit(f"- Feature Selection: <b>{'Enabled' if feature_selection else 'Disabled'}</b>")
            
            # Record start time for training duration calculation
            start_time = time.time()
            
            # Load data
            self.signals.status_message.emit("Loading data...")
            train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
                data_dir=data_dir,
                batch_size=32,
                test_split=0.2,
                feature_selection=feature_selection,
            )
            
            # Create client with custom results callback
            self.signals.status_message.emit("Connecting to server...")
            self.client = AsyncXGBoostClient(
                train_dmatrix=train_dmatrix,
                valid_dmatrix=valid_dmatrix,
                num_train=num_train,
                num_val=num_val,
                num_local_round=num_local_rounds,
                params=get_xgboost_params(),
                async_interval=async_interval,
            )
            
            # Add tracking attributes to client
            self.client.training_complete = False
            self.client.in_final_evaluation = False
            self.client.accuracy_history = []  # Track accuracy over rounds for convergence
            
            # Original evaluate function
            original_evaluate = self.client.evaluate
            
            # Override evaluate to capture metrics
            def evaluate_wrapper(ins):
                # Mark that we're in evaluation phase
                self.client.in_final_evaluation = True
                
                result = original_evaluate(ins)
                
                # Calculate training time
                training_time = time.time() - start_time
                
                # Add additional metrics
                metrics = result.metrics
                metrics['training_time'] = training_time
                
                # Estimate model size
                if hasattr(self.client, 'local_model') and self.client.local_model is not None:
                    try:
                        # Get model size by saving it to a temporary buffer
                        import io
                        buffer = io.BytesIO()
                        self.client.local_model.save_model(buffer)
                        metrics['model_size'] = len(buffer.getvalue())
                    except Exception as model_size_error:
                        self.signals.log.emit(f"<span style='color:{WARNING_COLOR};'>Warning: Could not calculate model size: {str(model_size_error)}</span>")
                        # Use a default size estimate if calculation fails
                        metrics['model_size'] = 10240  # Default to 10KB
                
                # Store accuracy for convergence rate calculation
                if 'accuracy' in metrics:
                    self.client.accuracy_history.append(metrics['accuracy'])
                
                # Add confusion matrix data if not present (simulated if necessary)
                if not all(key in metrics for key in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
                    # Use validation data to calculate confusion matrix
                    try:
                        # Get predictions
                        y_pred = self.client.local_model.predict(valid_dmatrix) if hasattr(self.client, 'local_model') else None
                        if y_pred is not None:
                            y_true = valid_dmatrix.get_label()
                            y_pred_binary = np.round(y_pred)
                            
                            # Calculate confusion matrix
                            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
                            fp = np.sum((y_pred_binary == 1) & (y_true == 0))
                            tn = np.sum((y_pred_binary == 0) & (y_true == 0))
                            fn = np.sum((y_pred_binary == 0) & (y_true == 1))
                            
                            metrics['true_positives'] = int(tp)
                            metrics['false_positives'] = int(fp)
                            metrics['true_negatives'] = int(tn)
                            metrics['false_negatives'] = int(fn)
                            
                            # Calculate additional metrics if not present
                            if 'precision' not in metrics:
                                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                            if 'recall' not in metrics:
                                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                            if 'f1_score' not in metrics:
                                precision = metrics['precision']
                                recall = metrics['recall']
                                metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    except Exception as e:
                        self.signals.log.emit(f"<span style='color:{WARNING_COLOR};'>Warning: Could not calculate confusion matrix: {str(e)}</span>")
                        # Provide placeholder values if calculation fails
                        if 'true_positives' not in metrics: metrics['true_positives'] = 0
                        if 'false_positives' not in metrics: metrics['false_positives'] = 0
                        if 'true_negatives' not in metrics: metrics['true_negatives'] = 0
                        if 'false_negatives' not in metrics: metrics['false_negatives'] = 0
                        if 'precision' not in metrics: metrics['precision'] = 0
                        if 'recall' not in metrics: metrics['recall'] = 0
                        if 'f1_score' not in metrics: metrics['f1_score'] = 0
                
                self.signals.metrics.emit(metrics)
                return result
            
            # Original fit function
            original_fit = self.client.fit
            
            # Override fit to track progress
            def fit_wrapper(ins):
                # Update current round from config
                if "global_round" in ins.config:
                    self.client.current_round = int(ins.config["global_round"])
                result = original_fit(ins)
                return result
            
            # Replace client's methods
            self.client.evaluate = evaluate_wrapper
            self.client.fit = fit_wrapper
            
            # Start client
            self.signals.status_message.emit("Training in progress...")
            fl.client.start_client(server_address=server_address, client=self.client)
            
            # Mark training as complete
            self.client.training_complete = True
            
            self.signals.log.emit(f"<span style='color:{SUCCESS_COLOR}; font-weight:bold;'>Training completed successfully</span>")
            self.signals.status_message.emit("Training completed")
            
            # Force progress to 100% on successful completion
            self.progress_bar.setValue(100)
            self.progress_label.setText("100%")
            
            # Enable prediction if we have a training interface
            if hasattr(self, 'predict_button'):
                self.predict_button.setEnabled(True)
                self.signals.log.emit(f"<span style='color:{MAIN_COLOR};'>Prediction feature is now available</span>")
            
        except Exception as e:
            self.signals.log.emit(f"<span style='color:{ERROR_COLOR}; font-weight:bold;'>Error during training: {str(e)}</span>")
            self.signals.status_message.emit(f"Error: {str(e)}")
        finally:
            # Ensure we always mark training as complete
            if hasattr(self, 'client') and self.client:
                self.client.training_complete = True
            self.signals.finished.emit()

    def update_progress(self):
        if self.client and hasattr(self.client, 'current_round'):
            # Update progress based on client's current round and status
            if hasattr(self.client, '_stop_requested') and self.client._stop_requested:
                # If stopping was requested, show 100%
                progress = 100
            elif hasattr(self.client, 'training_complete') and self.client.training_complete:
                # If training is complete, show 100%
                progress = 100
            else:
                # Estimate progress based on current round
                # Assuming a typical federated learning process might have 5 rounds total
                estimated_total_rounds = 5
                progress = min(int(self.client.current_round * 100 / estimated_total_rounds), 99)
                
                # If we know we're in the final evaluation phase, show 99%
                if hasattr(self.client, 'in_final_evaluation') and self.client.in_final_evaluation:
                    progress = 99
            
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"{progress}%")

    def on_training_finished(self):
        """Handle the training finished signal"""
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        
        # Ensure progress bar shows 100% when training finishes
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_status_label.setText("Status: Completed")
        self.training_status_label.setStyleSheet(f"font-weight: bold; color: {SUCCESS_COLOR};")
        self.status_indicator.setText("Ready")
        self.status_indicator.setStyleSheet(f"""
            color: {SUCCESS_COLOR};
            font-weight: bold;
            padding: 8px;
        """)

    def stop_training(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.signals.log.emit(f"<span style='color:{WARNING_COLOR}; font-weight:bold;'>Stopping training...</span>")
            self.signals.status_message.emit("Stopping training...")
            
            # We can't directly stop the thread, but we can set a flag
            if self.client:
                # Set a flag on the client to stop
                self.client._stop_requested = True
                # Also mark as complete to update progress correctly
                self.client.training_complete = True
        
        # Show 100% progress when stopped
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_status_label.setText("Status: Stopped")
        self.training_status_label.setStyleSheet(f"font-weight: bold; color: {WARNING_COLOR};")
        self.status_indicator.setText("Stopped")
        self.status_indicator.setStyleSheet(f"""
            color: {WARNING_COLOR};
            font-weight: bold;
            padding: 8px;
        """)
        
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()

    def closeEvent(self, event):
        """Handle application close event properly to avoid the QPaintDevice warning"""
        # Stop any ongoing training
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.is_alive():
            if self.client:
                self.client._stop_requested = True
                self.client.training_complete = True
            
            # Wait briefly for the thread to respond to stop request
            self.signals.log.emit(f"<span style='color:{WARNING_COLOR};'>Application is closing, stopping training...</span>")
            self.worker_thread.join(timeout=0.5)  # Wait up to 0.5 seconds
        
        # Stop timer if running
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        
        # Clean up any logging handlers
        if hasattr(self, 'text_handler'):
            logger.removeHandler(self.text_handler)
            
        # Process pending events before closing to avoid QPaintDevice warning
        QApplication.processEvents()
        
        # Accept the close event
        event.accept()

    def create_prediction_area(self):
        """Create the prediction area with image upload and visualization"""
        layout = QVBoxLayout(self.prediction_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Image selection section
        input_group = QGroupBox("Image Selection")
        input_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Select an image file...")
        self.image_path_input.setReadOnly(True)
        
        browse_button = QPushButton("Browse")
        browse_button.setIcon(QIcon("image.png"))  # Add image icon if available
        browse_button.clicked.connect(self.browse_image)
        
        file_layout.addWidget(self.image_path_input)
        file_layout.addWidget(browse_button)
        input_layout.addLayout(file_layout)
        
        # Predict button
        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.setIcon(QIcon("predict.png"))  # Add icon if available
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)  # Only enable after training
        
        input_layout.addWidget(self.predict_button)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Result section
        result_group = QGroupBox("Prediction Results")
        result_layout = QHBoxLayout()
        
        # Left side: Original image and prediction result
        left_panel = QVBoxLayout()
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet(f"""
            border: 1px solid #bdc3c7;
            background-color: white;
            border-radius: 4px;
        """)
        
        self.prediction_result_label = QLabel("Prediction: N/A")
        self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_result_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
        """)
        
        self.prediction_confidence_label = QLabel("Confidence: N/A")
        self.prediction_confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.prediction_result_label)
        left_panel.addWidget(self.prediction_confidence_label)
        
        # Right side: Grad-CAM visualization
        right_panel = QVBoxLayout()
        
        self.gradcam_label = QLabel("Grad-CAM visualization will appear here")
        self.gradcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gradcam_label.setMinimumSize(300, 300)
        self.gradcam_label.setStyleSheet(f"""
            border: 1px solid #bdc3c7;
            background-color: white;
            border-radius: 4px;
        """)
        
        self.gradcam_explanation = QLabel(
            "Grad-CAM highlights the regions that influenced the model's prediction."
        )
        self.gradcam_explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gradcam_explanation.setWordWrap(True)
        
        right_panel.addWidget(self.gradcam_label)
        right_panel.addWidget(self.gradcam_explanation)
        
        # Add panels to layout
        result_layout.addLayout(left_panel)
        result_layout.addLayout(right_panel)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Add technical details section
        details_group = QGroupBox("Technical Details")
        details_layout = QVBoxLayout()
        
        self.prediction_details = QTextEdit()
        self.prediction_details.setReadOnly(True)
        self.prediction_details.setMaximumHeight(100)
        self.prediction_details.setPlaceholderText("Prediction details will appear here")
        
        details_layout.addWidget(self.prediction_details)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

    def browse_image(self):
        """Browse for image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_path_input.setText(file_path)
            
            # Display the selected image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                
                # Enable predict button if model is available
                if hasattr(self, 'client') and self.client and hasattr(self.client, 'local_model'):
                    self.predict_button.setEnabled(True)
                    self.statusBar.showMessage("Image loaded, ready for prediction", 3000)
                else:
                    self.statusBar.showMessage("Image loaded, but no trained model available. Train a model first.", 5000)
            else:
                self.image_label.setText("Failed to load image")

    def run_prediction(self):
        """Run prediction on the selected image"""
        if not hasattr(self, 'client') or not self.client or not hasattr(self.client, 'local_model'):
            self.statusBar.showMessage("No trained model available. Train a model first.", 5000)
            return
            
        image_path = self.image_path_input.text()
        if not image_path or not os.path.isfile(image_path):
            self.statusBar.showMessage("Please select a valid image file", 3000)
            return
            
        # Disable the button during prediction
        self.predict_button.setEnabled(False)
        self.statusBar.showMessage("Running prediction...", 3000)
        
        # Connect the signal to handle prediction results
        self.signals.prediction_result.connect(self.update_prediction_ui)
        
        # Run prediction in a thread
        prediction_thread = threading.Thread(
            target=self.process_prediction,
            args=(image_path,),
            daemon=True
        )
        prediction_thread.start()

    def process_prediction(self, image_path):
        """Process the image and run prediction"""
        try:
            # Load and preprocess the image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Open image and apply transformations
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0)  # Create a batch
            
            # Get the ensemble feature extractor
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ensemble_models = get_ensemble_feature_extractor()
            
            # Extract features using the ensemble
            features_list = []
            gradcam_data = None
            
            for model_name, model in ensemble_models.items():
                model.eval()
                with torch.no_grad():
                    features = model(input_batch.to(device))
                    features_list.append(features.cpu().numpy())
                
                # Use the first model for Grad-CAM
                if gradcam_data is None:
                    gradcam_data = self.compute_gradcam(model, input_batch.to(device), image)
            
            # Combine features from all models
            combined_features = np.hstack([features for features in features_list])
            
            # Create DMatrix for XGBoost
            dmatrix = xgb.DMatrix(combined_features)
            
            # Run prediction with the trained model
            prediction = self.client.local_model.predict(dmatrix)
            
            # Get prediction results
            probability = prediction[0]
            predicted_class = "Infected" if probability > 0.5 else "Normal"
            class_probability = probability if probability > 0.5 else 1 - probability
            
            # Emit signal with results to update UI on main thread
            self.signals.prediction_result.emit(predicted_class, class_probability, probability, gradcam_data)
            
            # Log details via signal to ensure thread safety
            self.signals.log.emit(f"<span style='color:{MAIN_COLOR};'>Prediction completed for {os.path.basename(image_path)}: {predicted_class} ({class_probability:.4f})</span>")
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            # Use signal to safely update log from worker thread
            self.signals.log.emit(f"<span style='color:{ERROR_COLOR};'>{error_msg}</span>")
            
            # Update UI on the main thread using invokeMethod
            QMetaObject.invokeMethod(self.prediction_details, "setText", 
                                    Qt.ConnectionType.QueuedConnection,
                                    Q_ARG(str, error_msg))
                                    
            QMetaObject.invokeMethod(self.statusBar, "showMessage", 
                                    Qt.ConnectionType.QueuedConnection,
                                    Q_ARG(str, error_msg),
                                    Q_ARG(int, 5000))
        finally:
            # Re-enable the predict button on the main thread
            QMetaObject.invokeMethod(self.predict_button, "setEnabled", 
                                    Qt.ConnectionType.QueuedConnection,
                                    Q_ARG(bool, True))

    def compute_gradcam(self, model, input_tensor, original_image):
        """Compute Grad-CAM visualization for the given model and input."""
        try:
            # Make sure input tensor requires gradient
            input_tensor = input_tensor.clone().detach()
            input_tensor.requires_grad = True

            # Forward pass
            features = model.features(input_tensor)
            
            # For DenseNet, use the final convolutional layer
            if hasattr(model, 'features') and hasattr(features, 'shape'):
                # Store gradients
                gradients = None
                
                # Register hook to capture gradients
                def save_gradients(grad):
                    nonlocal gradients
                    gradients = grad.detach().cpu().numpy()
                
                # Add hook to the last convolutional layer output
                features.register_hook(save_gradients)
                
                # Forward pass through remaining layers
                output = model.classifier(torch.nn.functional.adaptive_avg_pool2d(
                    features, (1, 1)).view(features.size(0), -1))
                
                # Get the index of the class with maximum score
                max_score_idx = output.argmax().item()
                
                # Backward pass (calculate gradients)
                model.zero_grad()
                output[0, max_score_idx].backward()
                
                # Process collected gradients
                if gradients is not None:
                    # Take the gradients of the output w.r.t. the last conv layer
                    pooled_gradients = np.mean(gradients[0], axis=(1, 2))
                    
                    # Get the feature maps from the last conv layer
                    activations = features.detach().cpu().numpy()[0]
                    
                    # Weight the channels by corresponding gradients
                    for i in range(len(pooled_gradients)):
                        activations[i, :, :] *= pooled_gradients[i]
                    
                    # Generate heatmap by averaging over channels
                    heatmap = np.mean(activations, axis=0)
                    
                    # Apply ReLU to the heatmap
                    heatmap = np.maximum(heatmap, 0)
                    
                    # Normalize the heatmap
                    if np.max(heatmap) > 0:  # Avoid division by zero
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                    
                    # Resize heatmap to match the original image size
                    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
                    
                    # Apply colormap to create visualization
                    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    
                    # Convert original image to numpy
                    original_np = np.array(original_image)
                    # Convert RGB to BGR for OpenCV
                    original_np = original_np[:, :, ::-1].copy()
                    
                    # Overlay heatmap on original image
                    superimposed = cv2.addWeighted(original_np, 0.6, cam, 0.4, 0)
                    
                    # Convert back to RGB for display
                    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                    
                    return {
                        'heatmap': cam,
                        'superimposed': superimposed,
                        'cam': cam
                    }
        
        except Exception as e:
            # Use signals to emit log message from non-UI thread
            self.signals.log.emit(f"<span style='color:{WARNING_COLOR};'>Warning: Could not compute Grad-CAM: {str(e)}</span>")
            return None
            
        return None

    def update_prediction_ui(self, predicted_class, class_probability, raw_probability, gradcam_data):
        """Update UI with prediction results - called on the main thread via signals"""
        # Set prediction result with appropriate color
        color = SUCCESS_COLOR if predicted_class == "Normal" else WARNING_COLOR
        self.prediction_result_label.setText(f"Prediction: {predicted_class}")
        self.prediction_result_label.setStyleSheet(f"""
            font-size: 16px; 
            font-weight: bold; 
            color: {color}; 
            padding: 10px;
        """)
        
        # Set confidence
        self.prediction_confidence_label.setText(f"Confidence: {class_probability:.2%}")
        
        # Set technical details
        details = (
            f"Raw probability: {raw_probability:.6f}\n"
            f"Threshold: 0.5\n"
            f"Model: XGBoost with Ensemble Feature Extraction"
        )
        self.prediction_details.setText(details)
        
        # Update Grad-CAM visualization if available
        if gradcam_data:
            height, width, channel = gradcam_data['superimposed'].shape
            bytes_per_line = 3 * width
            
            q_image = QImage(
                gradcam_data['superimposed'].data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format.Format_RGB888
            )
            
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.gradcam_label.setPixmap(scaled_pixmap)
        else:
            self.gradcam_label.setText("Grad-CAM visualization not available")
            
        # Update status bar
        self.statusBar.showMessage(f"Prediction completed: {predicted_class} with {class_probability:.2%} confidence", 5000)
        
        # Disconnect the signal to avoid memory leaks
        self.signals.prediction_result.disconnect(self.update_prediction_ui)


if __name__ == "__main__":
    # Check and fix import error in QtGui
    if "QtPainter" in dir():
        # This is an invalid import that needs to be fixed
        from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
        # Remove the invalid QtPainter reference
    
    app = QApplication(sys.argv)
    
    # Set application-wide tooltips style
    QToolTip.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet(f"""
        QToolTip {{
            border: 1px solid {SECONDARY_COLOR};
            padding: 5px;
            border-radius: 3px;
            background-color: {SECONDARY_COLOR};
            color: white;
        }}
    """)
    
    # Clean exit handler to avoid QPaintDevice warning
    def clean_exit():
        QApplication.processEvents()
    
    # Connect clean exit handler
    app.aboutToQuit.connect(clean_exit)
    
    window = FederatedLearningApp()
    window.show()
    sys.exit(app.exec())