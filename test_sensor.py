import serial
import serial.tools.list_ports
import time
import sys

print("--- Arduino Sensor Test ---")
print("Scanning COM ports...")

try:
    ports = list(serial.tools.list_ports.comports())
except Exception as e:
    print(f"Error scanning ports: {e}")
    ports = []

arduino_port = None

if not ports:
    print("No COM ports found! Check your USB cable.")
else:
    for p in ports:
        print(f"Found: {p.device} - {p.description}")
        # Try to auto-select
        if "Arduino" in p.description or "CH340" in p.description or "USB Serial" in p.description or "Serial" in p.description:
            arduino_port = p.device

if arduino_port:
    print(f"\nAttempting to connect to: {arduino_port}")
    try:
        ser = serial.Serial(arduino_port, 115200, timeout=1)
        print("Connected! Waiting for data...")
        print("(Press Ctrl+C to stop)")
        
        start_time = time.time()
        while True:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"DATA: {line}")
                except Exception as e:
                    print(f"Read Error: {e}")
            
            # Timeout check
            if time.time() - start_time > 10 and not ser.in_waiting:
                # print("Waiting for data...") 
                pass
                
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Connection Error: {e}")
else:
    print("\n[ERROR] Could not identify Arduino port automatically.")
    print("Please check if the Arduino is plugged in.")
