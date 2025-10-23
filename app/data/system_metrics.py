# app/data/system_metrics.py

import psutil
import time
import json
from app.utils import logger # Assuming logger is correctly set up

TARGET_SERVER_IP = "10.200.2.192:7077"

def get_priority_status(metric_type: str, percent: float) -> str:
    """Calculates a dynamic priority based on the metric percentage."""
    if metric_type == "disk_free":
        if percent <= 10.0:
            return "critical"
        elif percent <= 20.0:
            return "warning"
        else:
            return "normal"
    elif metric_type == "cpu_used":
        if percent > 85.0:
            return "critical"
        elif percent > 70.0:
            return "warning"
        else:
            return "normal"
    elif metric_type == "memory_used":
        if percent > 90.0:
            return "critical"
        elif percent > 80.0:
            return "warning"
        else:
            return "normal"
    return "unknown"

def get_realtime_data_in_target_format():
    """Gathers real-time system metrics and formats them into the desired JSON structure."""
    current_time_str = time.strftime("%b %d, %Y %H:%M:%S")

    # Initialize defaults for safety
    drive_available_value = "Available"
    drive_available_priority = "normal"
    disk_free_mb = 0
    disk_used_mb = 0
    disk_free_percent = 0.0
    mem_used_mb = 0
    cpu_percent = 0.0
    mem_percent = 0.0

    try:
        # --- Collect Raw Metrics ---
        disk = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None) 

        # Calculate disk metrics
        disk_total_mb = disk.total / (1024 * 1024)
        disk_used_mb = round(disk.used / (1024 * 1024))
        disk_free_mb = round(disk.free / (1024 * 1024))
        disk_free_percent = disk.free / disk.total * 100
        
        # Calculate memory metrics
        mem_used_mb = round(memory.used / (1024 * 1024))
        mem_percent = memory.percent

    except Exception as e:
        logger.error(f"Error reading disk metrics via psutil: {e}")
        # --- DRIVE UNAVAILABILITY LOGIC ---
        drive_available_value = "Unavailable"
        drive_available_priority = "critical"
        # Since metrics could not be read, percentages remain at 0 for fallback priority logic

    # --- PRIORITY CALCULATIONS ---
    disk_priority = get_priority_status("disk_free", disk_free_percent) if drive_available_value == "Available" else drive_available_priority
    cpu_priority = get_priority_status("cpu_used", cpu_percent)
    mem_priority = get_priority_status("memory_used", mem_percent)

    # 4. Format the final JSON to match the image structure
    metrics_json = {
        f"Measure Details for {TARGET_SERVER_IP}:eG Manager": {
            "Disk Space": {
                "lastMeasurementTime": current_time_str,
                "C": [
                    # 1. Used space metric
                    {"Used space ": [
                        {"unit": "MB", 
                         "priority": mem_priority, 
                         "value": f"{mem_used_mb}"
                        }
                    ]},
                    # 2. Drive availability metric (DYNAMIC!)
                    {"Drive availability ": [
                        {"unit": "-", 
                         "priority": drive_available_priority,
                         "value": drive_available_value
                        }
                    ]},
                    # 3. Free space metric
                    {"Free space ": [
                        {"unit": "MB", 
                         "priority": disk_priority,
                         "value": f"{disk_free_mb}"
                        }
                    ]}
                ]
            },
            # --- Custom entries for RAG questions ---
            "CPU Utilization": {
                "lastMeasurementTime": current_time_str,
                "value": f"{cpu_percent}",
                "unit": "%",
                "priority": cpu_priority
            },
            "Memory Usage": {
                "lastMeasurementTime": current_time_str,
                "value": f"{mem_used_mb}",
                "unit": "MB",
                "priority": mem_priority
            }
        }
    }
    return metrics_json