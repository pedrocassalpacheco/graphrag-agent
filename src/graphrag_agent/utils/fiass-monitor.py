# monitor.py
import psutil
import time
import subprocess
import sys
import csv
from datetime import datetime

# Start your script
process = subprocess.Popen([sys.executable, "src/graphrag_agent/fiass-test.py"])
pid = process.pid

# Create CSV file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"memory_profile_{timestamp}.csv"

print(f"Monitoring PID {pid}...")
print(f"Saving data to: {csv_filename}")
print("Time(s)\tRSS(MB)\tVMS(MB)\t%MEM\tCPU%(1s)\tCPU_Time(s)\tStatus")
print("-" * 70)

start_time = time.time()
process_start_time = None

# Open CSV file for writing
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header with units
    csv_writer.writerow(
        [
            "Timestamp",
            "Elapsed_Seconds",
            "RSS_MB",
            "VMS_MB",
            "Memory_Percent",
            "CPU_Percent_1s",
            "CPU_Time_Seconds",
            "Status",
        ]
    )

    try:
        proc = psutil.Process(pid)

        # Get initial process creation time for CPU time calculation
        process_start_time = proc.create_time()

        while process.poll() is None:  # While process is still running
            try:
                memory_info = proc.memory_info()
                memory_percent = proc.memory_percent()
                cpu_percent = proc.cpu_percent(interval=None)  # Non-blocking
                cpu_times = proc.cpu_times()
                status = proc.status()

                # Calculate total CPU time (user + system)
                total_cpu_time = cpu_times.user + cpu_times.system

                elapsed = time.time() - start_time
                rss_mb = memory_info.rss / 1024 / 1024
                vms_mb = memory_info.vms / 1024 / 1024
                current_time = datetime.now().isoformat()

                # Print to console with units
                print(
                    f"{elapsed:.1f}s\t{rss_mb:.1f}MB\t{vms_mb:.1f}MB\t{memory_percent:.1f}%\t{cpu_percent:.1f}%\t{total_cpu_time:.2f}s\t{status}"
                )

                # Write to CSV
                csv_writer.writerow(
                    [
                        current_time,
                        f"{elapsed:.1f}",
                        f"{rss_mb:.1f}",
                        f"{vms_mb:.1f}",
                        f"{memory_percent:.1f}",
                        f"{cpu_percent:.1f}",
                        f"{total_cpu_time:.2f}",
                        status,
                    ]
                )
                csvfile.flush()  # Ensure data is written immediately

                time.sleep(0.5)  # Monitor every 500ms

            except psutil.NoSuchProcess:
                print("Process ended")
                break
            except psutil.AccessDenied:
                print("Access denied to process information")
                break

    except KeyboardInterrupt:
        print(f"\nStopping monitoring... Data saved to {csv_filename}")
        process.terminate()

print(f"Monitoring complete. Data saved to: {csv_filename}")

# Print summary
print("\nSummary of metrics:")
print("RSS(MB): Resident Set Size - Physical memory currently in RAM")
print("VMS(MB): Virtual Memory Size - Total virtual memory allocated")
print("%MEM: Percentage of total system memory being used")
print("CPU%(1s): CPU usage percentage over last measurement interval")
print("CPU_Time(s): Total CPU time consumed since process start (user + system)")
