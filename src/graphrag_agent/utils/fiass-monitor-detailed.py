# monitor-detailed.py
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
csv_filename = f"detailed_profile_{timestamp}.csv"

print(f"Monitoring PID {pid} - Detailed Memory & CPU Metrics")
print(f"Saving data to: {csv_filename}")
print(
    "Time(s)\tRSS(MB)\tVMS(MB)\t%MEM\tCPU%(1s)\tUser_CPU(s)\tSys_CPU(s)\tTotal_CPU(s)\tStatus"
)
print("-" * 90)

start_time = time.time()

# Open CSV file for writing
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write detailed header
    csv_writer.writerow(
        [
            "Timestamp",
            "Elapsed_Seconds",
            "RSS_MB",
            "VMS_MB",
            "Memory_Percent",
            "CPU_Percent_1s",
            "User_CPU_Time_Seconds",
            "System_CPU_Time_Seconds",
            "Total_CPU_Time_Seconds",
            "Status",
            "Memory_Change_MB",
        ]
    )

    previous_rss = 0

    try:
        proc = psutil.Process(pid)

        while process.poll() is None:  # While process is still running
            try:
                memory_info = proc.memory_info()
                memory_percent = proc.memory_percent()
                cpu_percent = proc.cpu_percent(interval=None)
                cpu_times = proc.cpu_times()
                status = proc.status()

                elapsed = time.time() - start_time
                rss_mb = memory_info.rss / 1024 / 1024
                vms_mb = memory_info.vms / 1024 / 1024
                user_cpu_time = cpu_times.user
                system_cpu_time = cpu_times.system
                total_cpu_time = user_cpu_time + system_cpu_time
                memory_change = rss_mb - previous_rss
                current_time = datetime.now().isoformat()

                # Print to console with all metrics and units
                print(
                    f"{elapsed:.1f}s\t{rss_mb:.1f}MB\t{vms_mb:.1f}MB\t{memory_percent:.1f}%\t"
                    f"{cpu_percent:.1f}%\t{user_cpu_time:.2f}s\t{system_cpu_time:.2f}s\t"
                    f"{total_cpu_time:.2f}s\t{status}"
                )

                # Write detailed data to CSV
                csv_writer.writerow(
                    [
                        current_time,
                        f"{elapsed:.1f}",
                        f"{rss_mb:.1f}",
                        f"{vms_mb:.1f}",
                        f"{memory_percent:.1f}",
                        f"{cpu_percent:.1f}",
                        f"{user_cpu_time:.2f}",
                        f"{system_cpu_time:.2f}",
                        f"{total_cpu_time:.2f}",
                        status,
                        f"{memory_change:.1f}",
                    ]
                )
                csvfile.flush()

                previous_rss = rss_mb
                time.sleep(0.5)

            except psutil.NoSuchProcess:
                print("Process ended")
                break
            except psutil.AccessDenied:
                print("Access denied")
                break

    except KeyboardInterrupt:
        print(f"\nStopping monitoring... Data saved to {csv_filename}")
        process.terminate()

print(f"\nMonitoring complete. Data saved to: {csv_filename}")

# Print detailed explanation
print("\n" + "=" * 60)
print("METRICS EXPLANATION:")
print("=" * 60)
print("RSS(MB):        Resident Set Size - Physical RAM currently used")
print("VMS(MB):        Virtual Memory Size - Total virtual memory allocated")
print("%MEM:           Percentage of total system RAM being used")
print("CPU%(1s):       CPU usage percentage over last measurement interval")
print("User_CPU(s):    CPU time spent in user mode (your code)")
print("Sys_CPU(s):     CPU time spent in system/kernel mode")
print("Total_CPU(s):   Total CPU time consumed (user + system)")
print("Status:         Process status (running, sleeping, etc.)")
