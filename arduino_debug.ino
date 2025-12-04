#include "MAX30100.h"
#include <Wire.h>


MAX30100 sensor;

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing MAX30100...");

  // Initialize the sensor
  if (!sensor.begin()) {
    Serial.println("FAILED to find MAX30100");
    while (1)
      ;
  }
  Serial.println("SUCCESS: MAX30100 found");

  // Set up the sensor for HR and SpO2
  sensor.setMode(MAX30100_MODE_SPO2_HR);

  // Set LED current (brightness).
  // If you get 0 values, try increasing this.
  // Options: MAX30100_LED_CURR_27MA, MAX30100_LED_CURR_50MA
  sensor.setLedsCurrent(MAX30100_LED_CURR_50MA, MAX30100_LED_CURR_50MA);
}

void loop() {
  uint16_t ir, red;

  sensor.update();

  while (sensor.getRawValues(&ir, &red)) {
    Serial.print("IR:");
    Serial.print(ir);
    Serial.print(" RED:");
    Serial.println(red);
  }
}
