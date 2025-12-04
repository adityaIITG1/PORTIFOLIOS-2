# Heart Rate Monitor Integration Guide

This guide helps you add live heart rate monitoring to your Yoga AI application using an Arduino Uno and a MAX30100 sensor.

## 1. Hardware Connections

Connect your MAX30100 sensor to the Arduino Uno as follows:

| MAX30100 Pin | Arduino Uno Pin | Note |
|--------------|-----------------|------|
| **VIN**      | **3.3V**        | **Important**: Use 3.3V. 5V might damage some modules unless they have a regulator. |
| **GND**      | **GND**         | Ground |
| **SCL**      | **A5**          | I2C Clock |
| **SDA**      | **A4**          | I2C Data |
| **INT**      | *Not Connected* | Optional |

> **Note**: If your MAX30100 module has a built-in voltage regulator (often has 5V and 3.3V pads), you can connect VIN to 5V. If unsure, stick to 3.3V.

## 2. Arduino Setup

1.  **Install Library**:
    *   Open Arduino IDE.
    *   Go to **Sketch** -> **Include Library** -> **Manage Libraries...**
    *   Search for **"MAX30100lib"** or **"MAX30100"**.
    *   Install the library by **OXullo Intersecans**.

2.  **Upload Code**:
    *   Open the file `arduino_heart_rate.ino` (located in your project folder) in Arduino IDE.
    *   Select your Board (**Arduino Uno**) and Port.
    *   Click **Upload**.
    *   Open the **Serial Monitor** (Tools -> Serial Monitor) and set baud rate to **115200**.
    *   You should see "Initializing..." and then "HR:..." data when you place your finger on the sensor.

## 3. Python Setup

You need to install the `pyserial` library to allow Python to talk to the Arduino.

1.  Open your terminal or command prompt.
2.  Run the following command:
    ```bash
    pip install pyserial
    ```

## 4. Running the Application

1.  Keep the Arduino connected via USB.
2.  Run your main Python script:
    ```bash
    python main1.py
    ```
3.  The application will automatically detect the Arduino.
4.  You will see a new **Heart Rate Panel** on the left side of the screen.
5.  **Biofeedback Insights**:
    *   **Green**: Deep Relaxation (Heart Rate < 70 bpm) - *Alpha Brain Waves*
    *   **Yellow**: Balanced/Focused (Heart Rate 70-90 bpm) - *Alpha-Beta Waves*
    *   **Red**: High Arousal (Heart Rate > 90 bpm) - *Beta-Gamma Waves*

## Troubleshooting

*   **"Sensor: Not Connected"**: Ensure the Arduino is plugged in and no other program (like Arduino Serial Monitor) is using the port.
*   **"FAILED" in Arduino Serial Monitor**: Check your wiring (SDA/SCL) and ensure the sensor is powered correctly.
