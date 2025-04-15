import sys
import os
import logging
import threading
import time
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, 
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer

# Import the client functions
from client import (
    load_data, get_xgboost_params, AsyncXGBoostClient, logger,
    save_model_with_timestamp
)
import flwr as fl
import xgb

# Create a custom logger for the GUI
class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        # Auto-scroll to the bottom
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


# Signal class for thread communication
class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    metrics = pyqtSignal(dict)


class FederatedLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Federated Learning Client")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create form for input parameters
        self.create_input_form()
        
        # Create log area
        self.create_log_area()
        
        # Create metrics display area
        self.create_metrics_area()
        
        # Create control buttons
        self.create_control_buttons()
        
        # Initialize variables
        self.client = None
        self.worker_thread = None
        self.signals = WorkerSignals()
        
        # Connect signals
        self.signals.log.connect(self.update_log)
        self.signals.metrics.connect(self.update_metrics)
        
        # Set up custom logger
        self.setup_logger()

    def create_input_form(self):
        form_group = QGroupBox("Configuration")
        form_layout = QFormLayout()
        
        # Data directory input
        self.data_dir_layout = QHBoxLayout()
        self.data_dir_input = QLineEdit()
        self.data_dir_button = QPushButton("Browse...")
        self.data_dir_button.clicked.connect(self.browse_data_dir)
        self.data_dir_layout.addWidget(self.data_dir_input)
        self.data_dir_layout.addWidget(self.data_dir_button)
        form_layout.addRow("Data Directory:", self.data_dir_layout)
        
        # Server address input
        self.server_address_input = QLineEdit("127.0.0.1:8080")
        form_layout.addRow("Server Address:", self.server_address_input)
        
        # Number of local rounds
        self.local_rounds_input = QSpinBox()
        self.local_rounds_input.setMinimum(1)
        self.local_rounds_input.setMaximum(100)
        self.local_rounds_input.setValue(1)
        form_layout.addRow("Local Rounds:", self.local_rounds_input)
        
        # Async interval
        self.async_layout = QHBoxLayout()
        self.async_checkbox = QCheckBox("Enable Async")
        self.async_interval_input = QSpinBox()
        self.async_interval_input.setMinimum(1)
        self.async_interval_input.setMaximum(3600)
        self.async_interval_input.setValue(30)
        self.async_interval_input.setEnabled(False)
        self.async_checkbox.toggled.connect(self.async_interval_input.setEnabled)
        self.async_layout.addWidget(self.async_checkbox)
        self.async_layout.addWidget(QLabel("Interval (seconds):"))
        self.async_layout.addWidget(self.async_interval_input)
        self.async_layout.addStretch()
        form_layout.addRow("Async Mode:", self.async_layout)
        
        # Feature selection
        self.feature_selection_checkbox = QCheckBox()
        form_layout.addRow("Enable Feature Selection:", self.feature_selection_checkbox)
        
        form_group.setLayout(form_layout)
        self.main_layout.addWidget(form_group)

    def create_log_area(self):
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(log_group)

    def create_metrics_area(self):
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Training Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)
        metrics_layout.addLayout(self.progress_layout)
        
        # Metrics display
        self.metrics_form = QFormLayout()
        self.accuracy_label = QLabel("N/A")
        self.auc_label = QLabel("N/A")
        self.training_time_label = QLabel("N/A")
        self.model_size_label = QLabel("N/A")
        
        self.metrics_form.addRow("Accuracy:", self.accuracy_label)
        self.metrics_form.addRow("AUC:", self.auc_label)
        self.metrics_form.addRow("Training Time:", self.training_time_label)
        self.metrics_form.addRow("Model Size:", self.model_size_label)
        metrics_layout.addLayout(self.metrics_form)
        
        metrics_group.setLayout(metrics_layout)
        self.main_layout.addWidget(metrics_group)

    def create_control_buttons(self):
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        
        self.main_layout.addLayout(buttons_layout)

    def setup_logger(self):
        # Set up logging to text widget
        self.text_handler = QTextEditLogger(self.log_text)
        self.text_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Add handler to logger
        logger.addHandler(self.text_handler)
        logger.setLevel(logging.INFO)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir_input.setText(dir_path)

    def update_log(self, message):
        self.log_text.append(message)

    def update_metrics(self, metrics):
        if 'accuracy' in metrics:
            self.accuracy_label.setText(f"{metrics['accuracy']:.4f}")
        
        # Look for AUC metric (it might have different names)
        for key in metrics:
            if 'auc' in key.lower():
                self.auc_label.setText(f"{metrics[key]:.4f}")
                break
        
        if 'training_time' in metrics:
            self.training_time_label.setText(f"{metrics['training_time']:.2f} seconds")
        
        if 'model_size' in metrics:
            size_kb = metrics['model_size'] / 1024
            self.model_size_label.setText(f"{size_kb:.2f} KB")

    def start_training(self):
        # Validate inputs
        data_dir = self.data_dir_input.text()
        if not os.path.isdir(data_dir):
            self.update_log("Error: Invalid data directory")
            return
        
        server_address = self.server_address_input.text()
        if not server_address:
            self.update_log("Error: Server address is required")
            return
        
        # Get other parameters
        num_local_rounds = self.local_rounds_input.value()
        async_interval = self.async_interval_input.value() if self.async_checkbox.isChecked() else None
        feature_selection = self.feature_selection_checkbox.isChecked()
        
        # Disable start button and enable stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Reset metrics
        self.accuracy_label.setText("N/A")
        self.auc_label.setText("N/A")
        self.training_time_label.setText("N/A")
        self.model_size_label.setText("N/A")
        
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

    def run_training(self, data_dir, server_address, num_local_rounds, async_interval, feature_selection):
        try:
            self.signals.log.emit(f"Starting training with parameters:")
            self.signals.log.emit(f"- Data Directory: {data_dir}")
            self.signals.log.emit(f"- Server Address: {server_address}")
            self.signals.log.emit(f"- Local Rounds: {num_local_rounds}")
            self.signals.log.emit(f"- Async Interval: {async_interval}")
            self.signals.log.emit(f"- Feature Selection: {feature_selection}")
            
            # Load data
            train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
                data_dir=data_dir,
                batch_size=32,
                test_split=0.2,
                feature_selection=feature_selection,
            )
            
            # Create client with custom results callback
            self.client = AsyncXGBoostClient(
                train_dmatrix=train_dmatrix,
                valid_dmatrix=valid_dmatrix,
                num_train=num_train,
                num_val=num_val,
                num_local_round=num_local_rounds,
                params=get_xgboost_params(),
                async_interval=async_interval,
            )
            
            # Original evaluate function
            original_evaluate = self.client.evaluate
            
            # Override evaluate to capture metrics
            def evaluate_wrapper(ins):
                result = original_evaluate(ins)
                self.signals.metrics.emit(result.metrics)
                return result
            
            # Replace client's evaluate method
            self.client.evaluate = evaluate_wrapper
            
            # Start client
            fl.client.start_client(server_address=server_address, client=self.client)
            
            self.signals.log.emit("Training completed successfully")
            
        except Exception as e:
            self.signals.log.emit(f"Error during training: {str(e)}")
        finally:
            self.signals.finished.emit()

    def update_progress(self):
        if self.client and hasattr(self.client, 'current_round'):
            # Update progress based on client's current round
            # This is just an estimate since we don't know the total number of rounds
            progress = min(self.client.current_round * 20, 100)  # Assume max 5 rounds
            self.progress_bar.setValue(progress)

    def stop_training(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.signals.log.emit("Stopping training...")
            # We can't directly stop the thread, but we can set a flag
            if self.client:
                # Set a flag on the client to stop
                self.client._stop_requested = True
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FederatedLearningApp()
    window.show()
    sys.exit(app.exec())