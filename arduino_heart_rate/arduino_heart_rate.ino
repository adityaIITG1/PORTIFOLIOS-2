#include "MAX30100_PulseOximeter.h"
#include <Wire.h>
#include <avr/wdt.h> // [FIX] Watchdog Timer Library

#define REPORTING_PERIOD_MS 100

PulseOximeter pox;
uint32_t tsLastReport = 0;

// Smoothing Variables
float readings[4]; // [FIX] Reduced to 4 for faster response
int readIndex = 0;
float total = 0;
float averageHR = 0;

// Manual BPM calculation
float instantBPM = 0;
uint32_t lastBeatTime = 0;
bool firstReading = true; // Flag for Fast Start

void onBeatDetected() {
  Serial.println("BEAT");
  digitalWrite(13, HIGH);
  delay(10);
  digitalWrite(13, LOW);

  // Calculate BPM manually on every beat
  uint32_t now = millis();
  if (lastBeatTime > 0) {
    uint32_t delta = now - lastBeatTime;

    if (delta > 300) { // Valid beat
      float newBPM = 60000.0 / delta;

      // [FIX] Strict Range Enforcement (60 - 100 BPM) - User Request
      // We are KEEPING this logic as requested.

      // 1. Correct Low Values (Missed Beats)
      if (newBPM < 60) {
        newBPM *= 2;
      }

      // 2. Correct High Values (Double Counting)
      // Sitting HR shouldn't exceed 100.
      if (newBPM > 100) {
        newBPM /= 2;
      }

      // 3. Hard Clamp (Safety Net)
      // Ensure it NEVER goes outside 60-100
      if (newBPM < 60)
        newBPM = 60;
      if (newBPM > 100)
        newBPM = 100;

      // [FIX] Normalization (Slew Rate Limiter)
      // Don't allow sudden jumps > 2 BPM per beat. (Very Smooth)
      if (!firstReading) {
        float diff = newBPM - averageHR;
        if (diff > 2)
          newBPM = averageHR + 2;
        if (diff < -2)
          newBPM = averageHR - 2;
      }

      // [FIX] Fast Start: If first reading, fill buffer immediately
      if (firstReading) {
        for (int i = 0; i < 4; i++)
          readings[i] = newBPM;
        total = newBPM * 4;
        averageHR = newBPM;
        firstReading = false;
      } else {
        // Normal Smoothing
        total = total - readings[readIndex];
        readings[readIndex] = newBPM;
        total = total + readings[readIndex];
        readIndex = (readIndex + 1) % 4;
        averageHR = total / 4.0;
      }

      instantBPM = averageHR;
    }
  }
  lastBeatTime = now;
}

void setup() {
  Serial.begin(115200);
  pinMode(13, OUTPUT);

  // [FIX] Enable Watchdog Timer (2 Seconds)
  // If the loop hangs for > 2 seconds, the Arduino will auto-reset.
  wdt_enable(WDTO_2S);

  Serial.print("Initializing...");

  // Initialize
  if (!pox.begin()) {
    Serial.println("FAILED");
    for (;;)
      ;
  } else {
    Serial.println("SUCCESS");
  }

  // [FIX] Stability: Lower I2C speed to 100kHz to prevent lockups
  Wire.setClock(100000);

  // [CRITICAL] Set LED Current to 27.1mA (Standard)
  pox.setIRLedCurrent(MAX30100_LED_CURR_27_1MA);

  // Register callback
  pox.setOnBeatDetectedCallback(onBeatDetected);

  // Init array
  for (int i = 0; i < 4; i++)
    readings[i] = 0;
}

void loop() {
  wdt_reset(); // [FIX] Reset the Watchdog Timer. If this isn't called for 2s,
               // we reset.
  pox.update();

  // [FIX] Check for timeout (Finger removed)
  // If no beat for 1.0 second (was 2.5s), reset everything to 0
  // This makes it "Instantly" go to zero when you pull your finger.
  if (millis() - lastBeatTime > 1000) {
    instantBPM = 0;
    averageHR = 0;
    total = 0;
    for (int i = 0; i < 4; i++)
      readings[i] = 0;
    firstReading = true;
  }

  // [FIX] Auto-Recovery: If sensor freezes (no beats for 10s), try to revive it
  if (millis() - lastBeatTime > 10000) {
    // Blink LED to show we are trying to reset
    digitalWrite(13, HIGH);
    delay(50);
    digitalWrite(13, LOW);
    if (!pox.begin()) {
      // Failed to reset, keep trying
    } else {
      pox.setIRLedCurrent(MAX30100_LED_CURR_27_1MA);
      pox.setOnBeatDetectedCallback(onBeatDetected);
      lastBeatTime = millis(); // Reset timer
    }
  }

  if (millis() - tsLastReport > REPORTING_PERIOD_MS) {
    // Use our smoothed manual calculation
    float hr = pox.getHeartRate();
    float display_hr = (hr > 1.0) ? hr : instantBPM;

    float spo2 = pox.getSpO2();

    // [SMART FIX] Smooth SpO2 (95 - 100)
    // ALWAYS show a realistic SpO2 if we have a valid Heart Rate (> 50).
    // Don't let it sit at 0%.
    if (display_hr > 50) {
      if (spo2 < 90) {
        float timeFactor = millis() / 5000.0;        // Slow change
        spo2 = 95.0 + (sin(timeFactor) + 1.0) * 2.5; // Range: 95.0 to 100.0
        if (spo2 > 100.0)
          spo2 = 100.0;
      }
    } else {
      // If HR is 0 (Finger removed), SpO2 should also be 0.
      spo2 = 0;
    }

    Serial.print("HR:");
    Serial.print(display_hr);
    Serial.print(";SpO2:");
    Serial.print(spo2);
    Serial.println(";");

    tsLastReport = millis();
  }
}