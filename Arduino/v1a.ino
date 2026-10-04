#include <Wire.h>
#include "Adafruit_TCS34725.h"

const int BUZZER_PIN = 8;

// 50 ms integration, 4x gain
Adafruit_TCS34725 rgbSensor(
  TCS34725_INTEGRATIONTIME_50MS,
  TCS34725_GAIN_4X
);

unsigned long lastRead = 0;
unsigned long lastZeroWarning = 0;
const unsigned long READ_INTERVAL = 250;   // FIX: the loop below now actually uses this (was a hard-coded 2500)

const uint16_t MIN_CLEAR = 50;
const float MIN_SATURATION = 0.20;

// detectColor() is unchanged. Python classifies the raw values with the same logic.
const char* detectColor(uint16_t r, uint16_t g, uint16_t b, uint16_t c) {
  if (c < MIN_CLEAR) return "black";

  // Normalize RGB so brightness has less effect
  float red = (float)r / c;
  float green = (float)g / c;
  float blue = (float)b / c;
  float maxVal = max(red, max(green, blue));
  float minVal = min(red, min(green, blue));
  float delta = maxVal - minVal;

  // Low saturation means white, gray, or a neutral color
  if (delta < 0.03) {
    if (c > 500) return "white";
    return "gray";
  }

  float saturation = delta / maxVal;

  if (saturation < MIN_SATURATION) return "gray";

  float hue;
  if (maxVal == red) {
    hue = 60.0 * ((green - blue) / delta);
    if (hue < 0) hue += 360.0;
  }
  else if (maxVal == green) {
    hue = 60.0 * ((blue - red) / delta + 2.0);
  }
  else {
    hue = 60.0 * ((red - green) / delta + 4.0);
  }

  if (hue < 15 || hue >= 345) return "red";
  if (hue < 45) return "orange";
  if (hue < 70) return "yellow";
  if (hue < 160) return "green";
  if (hue < 200) return "cyan";
  if (hue < 260) return "blue";
  if (hue < 290) return "purple";
  if (hue < 345) return "pink";

  return "unknown";
}

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);

  if (!rgbSensor.begin()) {
    Serial.println("ERROR: TCS34725 not detected");
    while (true) {
      delay(100);
    }
  }

  // FIX: if the I2C bus ever glitches, give up after 25 ms instead of freezing the whole
  // sketch (a frozen sketch looks exactly like "no sensor data"). Needs a recent AVR core;
  // the #if skips it on boards that don't support it.
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(25000, true);
#endif

  Serial.println("WRISTBAND_READY");
}

void loop() {
  // Receive commands from Python
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "BUZZER_ON") {
      tone(BUZZER_PIN, 2000);
      Serial.println("BUZZER_ON_OK");
    }
    else if (command == "BUZZER_OFF") {
      noTone(BUZZER_PIN);
      Serial.println("BUZZER_OFF_OK");
    }
    else if (command == "BEEP") {
      tone(BUZZER_PIN, 2000);
      delay(150);
      noTone(BUZZER_PIN);
    }
  }

  // Read the sensor every READ_INTERVAL ms
  if (millis() - lastRead >= READ_INTERVAL) {
    lastRead = millis();

    uint16_t r, g, b, c;
    rgbSensor.getRawData(&r, &g, &b, &c);

    // FIX: all four channels at 0 almost always means the sensor lost power or I2C wiring.
    // Say so (at most every 5 s) instead of silently sending zeros.
    if (r == 0 && g == 0 && b == 0 && c == 0 && millis() - lastZeroWarning > 5000) {
      lastZeroWarning = millis();
      Serial.println("ERROR: sensor returned all zeros (check SDA/SCL/power wiring)");
    }
    
    Serial.print("DATA,");
    Serial.print(r);
    Serial.print(",");
    Serial.print(g);
    Serial.print(",");
    Serial.print(b);
    Serial.print(",");
    Serial.println(c);
  }
}
